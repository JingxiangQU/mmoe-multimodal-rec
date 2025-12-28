from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
from collections import deque

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.io.fileio import WriteToFiles, destination_prefix_naming, TextSink
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.textio import ReadAllFromText, WriteToText
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


def parse_json(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logging.warning(f"Bad JSON, skip: {line[:200]}…")
        return None


def extract_main_image(imgs):
    """Return hi_res > large > thumb of first image entry."""
    if imgs and isinstance(imgs, list):
        im = imgs[0]
        return im.get("hi_res") or im.get("large") or im.get("thumb") or ""
    return ""


def flatten_strict(x):
    """Defensive flatten into either (dest, str) or ((cat, split), str)."""
    import json

    def to_str(v):
        if isinstance(v, str):
            return v
        if isinstance(v, bytes):
            return v.decode("utf-8")
        if isinstance(v, dict):
            return json.dumps(v)
        if isinstance(v, (tuple, list)):
            if not v:
                return ""
            return to_str(v[0])
        if isinstance(v, (int, float)):
            return str(v)
        return None

    if isinstance(x, tuple) and len(x) == 2:
        k, v = x
        v_str = to_str(v)
        if isinstance(k, tuple) and len(k) == 2 and all(isinstance(i, str) for i in k):
            if isinstance(v_str, str):
                return (k, v_str)
        elif isinstance(k, str):
            if isinstance(v_str, str):
                return (k, v_str)
        logging.critical(f"flatten_strict: key/value格式不符: {repr(x)}")
        return None

    if isinstance(x, tuple) and len(x) == 3 and all(isinstance(i, str) for i in x[:2]):
        v_str = to_str(x[2])
        if isinstance(v_str, str):
            return ((x[0], x[1]), v_str)
        logging.critical(f"flatten_strict: 三元组value不是str: {repr(x)}")
        return None

    logging.critical(f"flatten_strict: 未知结构: {repr(x)}")
    return None


def safe_addnl(kv):
    # ((cat, split), str)
    if isinstance(kv[0], tuple) and len(kv[0]) == 2:
        dest = f"{kv[0][1]}/{kv[0][0]}"
        return dest, kv[1] + "\n"
    # (dest, str)
    if isinstance(kv[0], str):
        return kv[0], kv[1] + "\n"
    logging.critical(f"AddNL遇到未知结构: {repr(kv)}")
    return None


class DownSampleByStar(beam.DoFn):
    """Keep only `rate_5` of 5★ review rows."""

    def __init__(self, rate_5: float = 0.2):
        self.rate_5 = rate_5

    def process(self, rec: dict):
        if int(rec.get("rating", 0)) == 5:
            if random.random() < self.rate_5:
                yield rec
        else:
            yield rec


class Enrich(beam.DoFn):
    """Join (parent_asin → [meta, review]) → positive sample rows."""

    def process(self, element):
        pid, grp = element
        if not grp["meta"] or not grp["review"]:
            return
        meta = grp["meta"][0]
        img_url = extract_main_image(meta.get("images", []))
        features = meta.get("features", [])
        description = meta.get("description", [])

        for rev in grp["review"]:
            rating = rev.get("rating")
            if rating is None:
                continue

            ts = rev.get("sort_timestamp")
            date_str = (
                datetime.datetime.utcfromtimestamp(ts / 1000).date().isoformat() if ts else None
            )

            yield {
                "user_id": rev["user_id"],
                "parent_asin": pid,
                "asin_child": rev.get("asin"),
                # meta
                "main_category": meta.get("main_category"),
                "product_title": meta.get("title"),
                "price": meta.get("price"),
                "main_image_url": img_url,
                "features": features,
                "description": description,
                # review content (for causal history)
                "review_title": rev.get("title", ""),
                "review_text": rev.get("text", ""),
                # event / label
                "sort_timestamp": ts,
                "event_date": date_str,
                "rating": rating,
                "label_good": 1 if rating >= 4 else 0,
                "label_best": 1 if rating == 5 else 0,
                "helpful_votes": rev.get("helpful_votes", 0),
                "_is_neg": 0,
            }


class CausalPosNegByUser(beam.DoFn):
    """Per-user causal user_feat (only past reviews) + negatives aligned to each positive row's time."""

    def __init__(self, neg_k: int):
        self.neg_k = int(neg_k)

    @staticmethod
    def _try_float(x):
        if x in (None, ""):
            return None
        try:
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _welford_update(n, mean, m2, x):
        n1 = n + 1
        delta = x - mean
        mean1 = mean + delta / n1
        delta2 = x - mean1
        m2_1 = m2 + delta * delta2
        return n1, mean1, m2_1

    @staticmethod
    def _welford_std(n, m2):
        if n <= 1:
            return 0.0
        return (m2 / (n - 1)) ** 0.5

    @staticmethod
    def _sample_k_not_seen(all_pids, seen: set, k: int, rnd: random.Random):
        if not all_pids:
            return []
        out = []
        tries = 0
        max_tries = max(200, k * 50)
        while len(out) < k and tries < max_tries:
            tries += 1
            pid = all_pids[rnd.randrange(0, len(all_pids))]
            if pid in seen or pid in out:
                continue
            out.append(pid)
        return out

    def process(self, element, all_pids):
        uid, rows_iter = element
        rows = list(rows_iter)

        def _ts_key(r):
            ts = r.get("sort_timestamp")
            return ts if isinstance(ts, (int, float)) else -1

        rows.sort(key=_ts_key)

        seen = set()
        cat_cnt = {}
        review_cnt = 0
        price_n, price_mean, price_m2 = 0, 0.0, 0.0
        hist = deque(maxlen=3)

        rnd = random.Random(hash(uid) & 0xFFFFFFFF)

        for r in rows:
            # build user_feat from PAST only
            if review_cnt <= 0:
                user_feat = {
                    "cat_hist": {},
                    "review_cnt": 0,
                    "price_mean": None,
                    "price_std": 0.0,
                    "history": [],
                }
            else:
                total = review_cnt
                user_feat = {
                    "cat_hist": {k: round(v / total, 4) for k, v in cat_cnt.items()},
                    "review_cnt": total,
                    "price_mean": round(price_mean, 4) if price_n > 0 else None,
                    "price_std": round(self._welford_std(price_n, price_m2), 4) if price_n > 1 else 0.0,
                    "history": list(hist),
                }

            # emit positive row
            r_pos = dict(r)
            r_pos["user_feat"] = user_feat
            r_pos["_is_neg"] = 0
            yield r_pos

            # update running state with current positive
            pid = r.get("parent_asin")
            if pid:
                seen.add(pid)

            cat = r.get("main_category") or "UNK"
            cat_cnt[cat] = cat_cnt.get(cat, 0) + 1
            review_cnt += 1

            px = self._try_float(r.get("price"))
            if px is not None:
                price_n, price_mean, price_m2 = self._welford_update(price_n, price_mean, price_m2, px)

            hist.append({"title": r.get("review_title", ""), "text": r.get("review_text", "")})

            # sample negatives aligned to this positive's time, reuse SAME causal user_feat
            for n_pid in self._sample_k_not_seen(all_pids, seen, self.neg_k, rnd):
                yield {
                    "user_id": uid,
                    "parent_asin": n_pid,
                    "label_good": 0,
                    "label_best": 0,
                    "rating": 0,
                    "helpful_votes": 0,
                    "sort_timestamp": r.get("sort_timestamp"),
                    "event_date": r.get("event_date"),
                    "user_feat": user_feat,
                    "_is_neg": 1,
                }


class AttachMetaNeg(beam.DoFn):
    """Attach meta to negatives. Preserve event_date and user_feat."""

    def process(self, element):
        pid, grp = element
        if not grp["meta"] or not grp["neg"]:
            return
        meta = grp["meta"][0]
        img_url = extract_main_image(meta.get("images", []))
        features = meta.get("features", [])
        description = meta.get("description", [])

        for n in grp["neg"]:
            yield {
                **n,
                "asin_child": None,
                "main_category": meta.get("main_category"),
                "product_title": meta.get("title"),
                "price": meta.get("price"),
                "main_image_url": img_url,
                "features": features,
                "description": description,
            }


class SplitByDate(beam.DoFn):
    def __init__(self, train_end: str, valid_end: str):
        self.t_end = datetime.date.fromisoformat(train_end)
        self.v_end = datetime.date.fromisoformat(valid_end)

    def process(self, element):
        row = json.loads(element) if isinstance(element, str) else element
        if not isinstance(row, dict):
            logging.critical(f"SplitByDate收到非dict: {repr(row)}")
            return

        split = "test"
        date_str = row.get("event_date")
        if date_str:
            d = datetime.date.fromisoformat(date_str)
            if d <= self.t_end:
                split = "train"
            elif d <= self.v_end:
                split = "valid"

        cat = row.get("main_category", "UNK")

        if "_is_neg" in row:
            row = {k: v for k, v in row.items() if k != "_is_neg"}

        yield (cat, split), json.dumps(row)


def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sports_meta", required=True)
    ap.add_argument("--sports_review", required=True)
    ap.add_argument("--tools_meta", required=True)
    ap.add_argument("--tools_review", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--valid_end", required=True)
    ap.add_argument("--neg_k", type=int, default=5)
    args, beam_args = ap.parse_known_args(argv)

    popts = PipelineOptions(beam_args, save_main_session=True)
    popts.view_as(SetupOptions).save_main_session = True

    meta_patterns = [args.sports_meta, args.tools_meta]
    review_patterns = [args.sports_review, args.tools_review]

    with beam.Pipeline(options=popts) as p:
        meta_pc = (
            p
            | "CreateMetaPatterns" >> beam.Create(meta_patterns)
            | "MatchMeta" >> fileio.MatchAll()
            | "ReadMetaText" >> ReadAllFromText(compression_type=CompressionTypes.GZIP)
            | "ParseMetaJSON" >> beam.Map(parse_json)
            | "FilterMetaValid" >> beam.Filter(lambda m: m and (m.get("parent_asin") or m.get("asin")))
            | "KV_meta" >> beam.Map(lambda m: (m.get("parent_asin") or m.get("asin"), m))
        )

        # image url dump
        _ = (
            meta_pc
            | "ToImgListKV" >> beam.Map(
                lambda kv: (
                    None,
                    json.dumps(
                        {
                            "parent_asin": kv[0],
                            "url": extract_main_image(kv[1].get("images", [])),
                            "main_category": kv[1].get("main_category", "UNK"),
                        }
                    ),
                )
            )
            | "ExtractValue" >> beam.Map(lambda kv: kv[1])
            | "WriteJsonl" >> WriteToText(
                file_path_prefix=args.output + "/image_urls",
                file_name_suffix=".jsonl",
                shard_name_template="-SS-of-NN",
            )
        )

        review_pc = (
            p
            | "CreateReviewPatterns" >> beam.Create(review_patterns)
            | "MatchReview" >> fileio.MatchAll()
            | "ReadReviewText" >> ReadAllFromText(compression_type=CompressionTypes.GZIP)
            | "ParseReviewJSON" >> beam.Map(parse_json)
            | "FilterReviewValid" >> beam.Filter(lambda r: r and (r.get("parent_asin") or r.get("asin")))
            | "Down5" >> beam.ParDo(DownSampleByStar())
            | "KV_rev" >> beam.Map(lambda r: (r.get("parent_asin") or r.get("asin"), r))
        )

        joined = {"meta": meta_pc, "review": review_pc} | "MetaReviewJoin" >> beam.CoGroupByKey()
        pos = joined | "EnrichPos" >> beam.ParDo(Enrich())

        all_pids_sampled = (
            meta_pc
            | "AllsampledKeys" >> beam.Keys()
            | "SampleAllPids" >> beam.combiners.Sample.FixedSizeGlobally(10000)
        )

        posneg = (
            pos
            | "KV_User" >> beam.Map(lambda r: (r["user_id"], r))
            | "GroupUser" >> beam.GroupByKey()
            | "CausalPosNeg" >> beam.ParDo(
                CausalPosNegByUser(args.neg_k),
                all_pids=beam.pvalue.AsSingleton(all_pids_sampled),
            )
        )

        pos_rows = posneg | "FilterPos" >> beam.Filter(lambda r: int(r.get("_is_neg", 0)) == 0)
        neg_rows = posneg | "FilterNeg" >> beam.Filter(lambda r: int(r.get("_is_neg", 0)) == 1)

        neg_full = (
            {"neg": neg_rows | "NegKV" >> beam.Map(lambda n: (n["parent_asin"], n)), "meta": meta_pc}
            | "JoinNegMeta" >> beam.CoGroupByKey()
            | "AttachMetaNeg" >> beam.ParDo(AttachMetaNeg())
        )

        all_rows = (pos_rows, neg_full) | "FlattenPosNeg" >> beam.Flatten()

        split_lines = (
            all_rows
            | "SplitByDate" >> beam.ParDo(SplitByDate(args.train_end, args.valid_end))
            | "FlattenBadStructures" >> beam.Map(flatten_strict)
            | "DropBad" >> beam.Filter(lambda x: x is not None)
            | "AddNL" >> beam.Map(safe_addnl)
            | "DropAddNLError" >> beam.Filter(lambda x: x is not None)
        )

        _ = (
            split_lines
            | "DropNotStrStr" >> beam.Filter(
                lambda x: x is not None and isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], str)
            )
            | "WriteRows" >> WriteToFiles(
                path=args.output,
                destination=lambda kv: kv[0],
                file_naming=destination_prefix_naming(suffix=".jsonl"),
                sink=lambda dest: TextSink(),
            )
        )


if __name__ == "__main__":
    run()

