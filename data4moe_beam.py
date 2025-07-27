

from __future__ import annotations
import argparse, json, random, statistics, datetime
import gzip,io
import json
import logging
import apache_beam as beam
from apache_beam.io.fileio import WriteToFiles, destination_prefix_naming, TextSink
from apache_beam.io import fileio
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io.textio import ReadAllFromText
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.textio import WriteToText




def parse_json(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logging.warning(f"Bad JSON, skip: {line[:200]}…")
        return None
    return json.loads(line)


def extract_main_image(imgs):
    """Return hi_res > large > thumb of first image entry."""
    if imgs and isinstance(imgs, list):
        im = imgs[0]
        return im.get("hi_res") or im.get("large") or im.get("thumb") or ""
    return ""

def flatten_strict(x):
    """
    展平为 (str, str) 或 ((str, str), str)；其它全部 log+None
    """
    import logging

    # 展开 value
    def to_str(v):
        if isinstance(v, str):
            return v
        if isinstance(v, bytes):
            return v.decode("utf-8")
        if isinstance(v, dict):
            return json.dumps(v)
        if isinstance(v, (tuple, list)):
            if not v: return ""
            return to_str(v[0])
        if isinstance(v, (int, float)):
            return str(v)
        return None

    # 展开 key，支持 (str, str) 或 str
    if isinstance(x, tuple) and len(x) == 2:
        k, v = x
        v_str = to_str(v)
        if isinstance(k, tuple) and len(k) == 2 and all(isinstance(i, str) for i in k):
            if isinstance(v_str, str):
                return (k, v_str)
        elif isinstance(k, str):
            if isinstance(v_str, str):
                return (k, v_str)
        # 其它全部 log+None
        logging.critical(f"flatten_strict: key/value格式不符: {repr(x)}")
        return None
    # 支持直接三元组 (a, b, v)
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
    # 其它结构丢弃
    import logging
    logging.critical(f"AddNL遇到未知结构: {repr(kv)}")
    return None
# ---------------------------------------------------------------------
# DoFns
# ---------------------------------------------------------------------

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
        features    = meta.get("features", [])      # list of str
        description = meta.get("description", [])   # list of str
        for rev in grp["review"]:
            rating = rev.get("rating")
            if rating is None:
                continue
            ts = rev.get("sort_timestamp")
            date_str = (datetime.datetime.utcfromtimestamp(ts/1000)
                        .date().isoformat()) if ts else None
            yield {
                "user_id": rev["user_id"],
                "parent_asin": pid,
                "asin_child": rev.get("asin"),
                "main_category": meta.get("main_category"),
                "title":          meta.get("title"),
                "price":          meta.get("price"),
                "main_image_url": img_url,
                "features":       features,
                "description":    description,
                # event / label / side features…
                "event_date":     date_str,
                "rating":         rating,
                "label_good":     1 if rating >= 4 else 0,
                "label_best":     1 if rating == 5 else 0,
                "helpful_votes":  rev.get("helpful_votes", 0),
            }



class UserProfile(beam.DoFn):
    def process(self, element):
        uid, rows = element
        rows = list(rows)
        cat_cnt, prices, votes = {}, [], []
        for r in rows:
            cat = r.get("main_category", "UNK")
            cat_cnt[cat] = cat_cnt.get(cat, 0) + 1
            if r.get("price") not in (None, ""):
                try:
                    prices.append(float(r["price"]))
                except Exception:
                    pass
            votes.append(int(r.get("helpful_votes", 0)))
        total = sum(cat_cnt.values()) or 1

        # 添加最多3条历史评论的title和text
        history = []
        for r in rows[:3]:
            history.append({
                "title": r.get("title", ""),
                "text": r.get("text", "")
            })

        prof = {
            "user_feat": {
                "cat_hist": {k: round(v / total, 4) for k, v in cat_cnt.items()},
                "review_cnt": total,
                "price_mean": round(statistics.mean(prices), 4) if prices else None,
                "price_std": round(statistics.stdev(prices), 4) if len(prices) > 1 else 0.0,
                # "helpful_votes_avg": ... 
                "history": history         
            }
        }
        yield uid, prof


class NegSampler(beam.DoFn):
    def __init__(self, k: int):
        self.k = k

    def process(self, element, all_pids):
        uid, seen = element
        candidates = list(set(all_pids) - set(seen))  # 这里 all_pids 就是已注入的真实 SideInput 对象
        random.shuffle(candidates)
        for pid in candidates[: self.k]:
            yield {
                "user_id": uid,
                "parent_asin": pid,
                "label_good": 0,
                "label_best": 0,
            }



class AttachMetaNeg(beam.DoFn):
    def process(self, element):
        pid, grp = element
        if not grp["meta"] or not grp["neg"]:
            return
        meta = grp["meta"][0]
        img_url = extract_main_image(meta.get("images", []))
        features    = meta.get("features", [])
        description = meta.get("description", [])
        for n in grp["neg"]:
            yield {
                **n,
                "main_category":   meta.get("main_category"),
                "title":           meta.get("title"),
                "price":           meta.get("price"),
                "main_image_url":  img_url,
            
                "features":        features,
                "description":     description,
                # 负样本没有 event_date
                "event_date":      None,
            }




class SplitByDate(beam.DoFn):
    def __init__(self, train_end: str, valid_end: str):
        # 这里用的是模块级的 datetime.date
        self.t_end = datetime.date.fromisoformat(train_end)
        self.v_end = datetime.date.fromisoformat(valid_end)

    def process(self, element):
        # 如果上游给的是 JSON 字符串，先反序列化
        if isinstance(element, str):
            row = json.loads(element)
        else:
            row = element
        if not isinstance(row, dict):
            import logging
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
        # 输出 ((cat, split), json_str)
        yield (cat, split), json.dumps(row)

# ---------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------

def run(argv=None):
    ap = argparse.ArgumentParser()
    #ap.add_argument("--bucket", required=True)
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

    meta_patterns = [
        args.sports_meta,
        args.tools_meta,
    ]
    review_patterns = [
        args.sports_review,
        args.tools_review,
    ]
    meta_pattern_str = ",".join(meta_patterns)
    review_pattern_str = ",".join(review_patterns)

    with beam.Pipeline(options=popts) as p:
        # 1. META ------------------------------------------------------------------

        meta_pc = (
            p
            | "CreateMetaPatterns" >> beam.Create(meta_patterns)
            | "MatchMeta"          >> fileio.MatchAll()
            # 用 ReadAllFromText 直接按行读取、自动解压 GZIP
            | "ReadMetaText"       >> ReadAllFromText(compression_type=CompressionTypes.GZIP)
            | "ParseMetaJSON"      >> beam.Map(parse_json)
            | "FilterMetaValid"    >> beam.Filter(lambda m: m and (m.get("parent_asin") or m.get("asin")))
            | "KV_meta"            >> beam.Map(lambda m: (m.get("parent_asin") or m.get("asin"), m))
        )
        # 1.1 image url list per category



        kv_records = (
            meta_pc
            | "ToImgListKV" >> beam.Map(lambda kv: (
                None,  
                json.dumps({
                    "parent_asin":  kv[0],
                    "url":           extract_main_image(kv[1].get("images", [])),
                    "main_category": kv[1].get("main_category","UNK"),
                })
            ))
        )

        lines = kv_records | "ExtractValue" >> beam.Map(lambda kv: kv[1])

        _ = (
            lines
            | "WriteJsonl" >> WriteToText(
                file_path_prefix=args.output + "/image_urls",
                file_name_suffix=".jsonl",
                shard_name_template="-SS-of-NN"
            )
        )
        review_pc = (
            p
            | "CreateReviewPatterns" >> beam.Create(review_patterns)
            | "MatchReview"          >> fileio.MatchAll()
            | "ReadReviewText"       >> ReadAllFromText(compression_type=CompressionTypes.GZIP)
            | "ParseReviewJSON"      >> beam.Map(parse_json)
            | "FilterReviewValid"    >> beam.Filter(lambda r: r and (r.get("parent_asin") or r.get("asin")))
            | "Down5"                >> beam.ParDo(DownSampleByStar())
            | "KV_rev"               >> beam.Map(lambda r: (r.get("parent_asin") or r.get("asin"), r))
        )

        # 3. JOIN & POSITIVE ----------------------------------------------------
        all_pids = meta_pc | "AllKeys" >> beam.Keys() | "AllToList" >> beam.combiners.ToList()
        all_pids_sampled = (
            meta_pc
            | "AllsampledKeys" >> beam.Keys()
            | "SampleAllPids" >> beam.combiners.Sample.FixedSizeGlobally(10000)
            # Sample后得到 [list]，不需要再 ToList
        )
        joined = {"meta": meta_pc, "review": review_pc} | "MetaReviewJoin" >> beam.CoGroupByKey()
        pos = joined | "EnrichPos" >> beam.ParDo(Enrich())

        # 4. USER PROFILE -------------------------------------------------------
        profiles = (
            pos
            | "MapUser" >> beam.Map(lambda r: (r["user_id"], r))
            | "GroupUser" >> beam.GroupByKey()
            | "UserProfile" >> beam.ParDo(UserProfile())
        )

        # 5. NEGATIVE -----------------------------------------------------------

        seen = (
            pos
            | "SeenUserProd" >> beam.Map(lambda r: (r["user_id"], r["parent_asin"]))
            | "GroupSeen"    >> beam.GroupByKey()
        )

        neg = (
            seen
            | "NegSampler" >> beam.ParDo(
                NegSampler(args.neg_k), 
                all_pids=beam.pvalue.AsSingleton(all_pids_sampled)  
            )
        )

        neg_kv = neg | "NegKV" >> beam.Map(lambda n: (n["parent_asin"], n))
        neg_full = (
            {"neg": neg_kv, "meta": meta_pc}
            | "JoinNegMeta" >> beam.CoGroupByKey()
            | "AttachMetaNeg" >> beam.ParDo(AttachMetaNeg())
        )

        # 6. UNION POS & NEG ----------------------------------------------------
        all_rows = (pos, neg_full) | "FlattenPosNeg" >> beam.Flatten()
        #all_rows = all_rows | "CheckAndFlattenRows" >> beam.Map(flatten_to_str)
        # 6.1 MERGE PROFILE -----------------------------------------------------
        with_prof = (
            {
                "row": all_rows | "UserRowKV" >> beam.Map(lambda r: (r["user_id"], r)),
                "prof": profiles,
            }
            | "JoinProfRow" >> beam.CoGroupByKey()
            #| "FlattenAfterCoGroup" >> beam.Map(flatten_to_str)
            | "MergeProfRowAll" >> beam.FlatMap(
                lambda kv: [
                    { **row, 
                    "user_feat": kv[1]["prof"][0]["user_feat"] if kv[1]["prof"] else {} 
                    }
                    for row in kv[1]["row"]
                ]
            )
        )

        # 7. SPLIT + WRITE ------------------------------------------------------

        split_lines = (
            with_prof
            | "SplitByDate" >> beam.ParDo(SplitByDate(args.train_end, args.valid_end))
            | "FlattenBadStructures" >> beam.Map(flatten_strict)  # 防御型展平
            | "DropBad" >> beam.Filter(lambda x:
                x is not None and
                (
                    (isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], tuple) and len(x[0]) == 2 and all(isinstance(i, str) for i in x[0]) and isinstance(x[1], str))
                    or
                    (isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], str))
                )
            )


            | "AddNL" >> beam.Map(safe_addnl)
            | "DropAddNLError" >> beam.Filter(lambda x: x is not None)
        )

        #split_lines | "DebugPrint" >> beam.Map(print)   # 检查输出

        _ = (
            split_lines
            | "FinalTypeCheck" >> beam.Map(lambda x: (
                logging.critical(f"WRITE前: {type(x)} {type(x[0])} {type(x[1])} {repr(x)}") if not (
                    x is not None and
                    isinstance(x, tuple) and len(x) == 2 and
                    isinstance(x[0], str) and isinstance(x[1], str)
                ) else None
            ) or x)
            | "DropNotStrStr" >> beam.Filter(lambda x:
                x is not None and
                isinstance(x, tuple) and len(x) == 2 and
                isinstance(x[0], str) and isinstance(x[1], str)
            )
            | "WriteRows" >> WriteToFiles(
                path=args.output,
                destination=lambda kv: kv[0],
                file_naming=destination_prefix_naming(suffix=".jsonl"),
                sink=lambda dest: TextSink()
            )
        )
if __name__ == "__main__":
    run()
