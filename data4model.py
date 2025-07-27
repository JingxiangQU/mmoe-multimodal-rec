"""
将 train / patch 合并并写 WebDataset tar 分片（GCS 直接可读）。
"""
import os, json, uuid, base64, argparse, typing as T
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import webdataset as wds
from apache_beam.io.filesystem import CompressionTypes
from google.cloud import storage  # 需要在 requirements.txt 加入 google-cloud-storage
import re
import unicodedata
import html
import emoji
from typing import Union, Sequence, List

# -------------------------------------------------
# 1. KV 解析函数
# -------------------------------------------------

def train2kv(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
        return (d["parent_asin"].strip(), d)
    except Exception:
        return None

def patch2kv(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
        return (d["parent_asin"].strip(), d)
    except Exception:
        return None
def smart_join(features):
    res = []
    for feat in features:
        feat = feat.strip()
        # 如果结尾已经是常见句子标点，则直接用
        if re.search(r'[。.;；.!?？！]$', feat):
            res.append(feat)
        else:
            res.append(feat + ';')
    return " ".join(res)
# -------------------------------------------------
# 2. 合并函数
# -------------------------------------------------


def normalize_text(*args: Union[str, Sequence[str]]) -> Union[str, List[str]]:
    """
    支持 normalize_text("a","b","c") 或 normalize_text(["a","b","c"])
    如果只传一个字符串，返回清洗后的单个 str；
    如果传多个字符串，返回 List[str]。
    """
    # 展平参数
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        texts = list(args[0])
    else:
        texts = list(args)

    def _clean_one(s: str) -> str:
        # --- 新增：先把所有 emoji 转成 :name: 形式 ---
        s = emoji.demojize(s, delimiters=(" ", " "))

        # 1) HTML 实体反转
        s = html.unescape(s)
        # 2) Unicode 规范化（兼容分解 + 合并）
        s = unicodedata.normalize("NFKC", s)
        # 3) 弯引号→直引号
        s = re.sub(r"[‘’‚‛❛❜\u2018\u2019\u201A\u201B]", "'", s)
        s = re.sub(r"[“”„‟❝❞\u201C\u201D\u201E\u201F]", '"', s)
        # 4) 破折号→短横线
        s = re.sub(r"[–—―\u2013\u2014\u2015]", "-", s)
        # 5) 省略号→三个点
        s = re.sub(r"[…\u2026]", "...", s)
        # 6) 换行/制表符→空格，剔除其它控制字符
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
        # 7) 处理反斜杠转义
        s = s.replace(r'\"', '"').replace(r"\'", "'")
        s = s.replace("\\", " ")
        # 8) 合并多余空白并去掉首尾
        s = re.sub(r"\s+", " ", s).strip()
        return s

    cleaned = [_clean_one(text) for text in texts]
    return cleaned[0] if len(cleaned) == 1 else cleaned


def build_user_text(rec: dict) -> str:
    uf = rec.get("user_feat", {})

    # 1. 类别分布
    cat_hist = {k: v for k, v in uf.get("cat_hist", {}).items() if v and v > 0}
    if cat_hist:
        cat_hist_str = "; ".join(f"{cat}: {cnt*100:.0f}%" for cat, cnt in cat_hist.items())
    else:
        cat_hist_str = "No browsing history"

    # 2. 评论次数
    review_cnt = uf.get("review_cnt", 0)
    review_cnt_str = f"{review_cnt}" if review_cnt > 0 else "No reviews"

    # 3. 价格统计
    price_mean = uf.get("price_mean")
    price_mean_str = f"{price_mean:.2f}" if price_mean is not None else "N/A"
    price_std = uf.get("price_std", 0.0)
    price_std_str = f"{price_std:.2f}" if price_std > 0 else "No price variation"

    # 4. 历史评论摘要

    raw_history = uf.get("history", [])
    parts = []
    for h in raw_history:
        piece = h.get("text") or h.get("title") or ""
        if piece:
            clean_piece = normalize_text(piece)
            parts.append(clean_piece)
    if parts:
        history_pieces = [f"Review{i+1}: {p}" for i, p in enumerate(parts)]
        history_str = smart_join(history_pieces)
        if not history_str.endswith((".", "!", "?", "。", ";", "；", "！", "？")):
            history_str += "."      
    else:
        history_str = "No review history."
    return (
        f"Category history: {cat_hist_str}. "
        f"Total reviews: {review_cnt_str}. "
        f"Avg price: {price_mean_str}. Price std: {price_std_str}. "
        f"Review history: {history_str}"
    )

def build_item_text(rec: dict) -> str:
    # 1. 主分类
    category = rec.get("main_category") or "Unknown category"
    # 2. 标题
    title = rec.get("title") or "No title"
    # 3. 价格
    price = rec.get("price")
    price_str = f"{price:.2f}" if price is not None else "N/A"

    # 4. 特征列表
    raw_feats = rec.get("features", [])
    clean_feats = normalize_text(raw_feats)  # 返回 List[str]
    if clean_feats:
        feats = smart_join(clean_feats)

       
        if not feats.endswith((".", "!", "?", "。", ";", "；", "！", "？")):
            feats += "."
        features_text = f"Item features: {feats}"
    else:
        features_text = "Item features: No features."

    # 5. 描述列表
    
    raw_descs = rec.get("description", []) or []  
    if isinstance(raw_descs, str):
        raw_descs = [raw_descs]
    clean_descs = normalize_text(raw_descs)

    if clean_descs:
        descs = smart_join(clean_descs)
        if not descs.endswith((".", "!", "?", "。", ";", "；", "！", "？")):
            descs += "."
        desc_text = f"Item description: {descs}"
    else:
        desc_text = "Item description: No description."

    return (
        f"Item category: {category}. "
        f"Item title: {title}. "
        f"Item price: {price_str}. "
        f"{features_text} "
        f"{desc_text}"
    )

def merge_patch(element: T.Tuple[str, dict]):
    """
    输入： (parent_asin, {"train": [...], "patch": [...]})
    输出：精简后的样本 dict，字段为：
      - key (str)
      - user_text (str), item_text (str)
      - patch_b64 (str), shape (list[int]), has_image (0/1)
      - label_good (int), label_best (int)
    """
    _, groups = element
    trains = groups.get("train", [])
    patches = groups.get("patch", [])
    patch_obj = patches[0] if patches else None

    for row in trains:
        user_id    = row.get("user_id", "unknown_user")
        parent_asin= row.get("parent_asin", "unknown_item")

        # 构造全局唯一 key
        key = f"{user_id}-{parent_asin}-{uuid.uuid4().hex[:6]}"

        # 三路文本
        user_text = build_user_text(row)
        item_text = build_item_text(row)

        # Patch
        if patch_obj:
            patch_b64 = patch_obj.get("patch_b64", "")
            shape     = patch_obj.get("shape", [196,3,16,16])
            has_image = 1
        else:
            patch_b64 = base64.b64encode(
                b"\x00" * (196 * 3 * 16 * 16 * 2)
            ).decode()
            shape     = [196,3,16,16]
            has_image = 0

        # 标签（保留，但不当作特征）
        label_good = int(row.get("label_good", 0))
        label_best = int(row.get("label_best", 0))

        yield {
            "key":        key,
            "user_text":  user_text,
            "item_text":  item_text,
            "patch_b64":  patch_b64,
            "shape":      shape,
            "has_image":  has_image,
            "label_good": label_good,
            "label_best": label_best
        }

# -------------------------------------------------
# 3. DoFn 写 WebDataset 分片
# -------------------------------------------------

class WriteWebDataset(beam.DoFn):
    def __init__(self, output_dir, compress=True):
        self.output_dir = output_dir.rstrip("/")
        self.compress = compress
        self._shard_index = 0

    def process(self, batch):
        # batch是List[dict]
        uniq = uuid.uuid4().hex[:8]
        suffix = ".tar.gz" if self.compress else ".tar"
       
        shard_path = os.path.join(self.output_dir,f"data-{self._shard_index:06d}-{uniq}{suffix}")
        with wds.TarWriter(shard_path, compress=self.compress) as sink:
            for rec in batch:
                key = rec.get("key") or f"{rec['parent_asin']}-{uuid.uuid4().hex[:8]}"
                sink.write({"__key__": key, "user.json": rec["user_text"].encode("utf-8")})
                sink.write({"__key__": key, "item.json": rec["item_text"].encode("utf-8")})
                sink.write({"__key__": key, "patch.bin": base64.b64decode(rec["patch_b64"])})
                sink.write({"__key__": key, "misc.json": json.dumps({"has_image": rec["has_image"], "shape": rec["shape"]}).encode("utf-8")})
                sink.write({"__key__": key, "label.json": json.dumps({"label_good": rec["label_good"], "label_best": rec["label_best"]}).encode("utf-8")})
        self._shard_index += 1


# -------------------------------------------------
# 4. 主函数
# -------------------------------------------------

def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    #parser.add_argument("--shard_size", type=int, default=10_000)
    args, beam_args = parser.parse_known_args(argv)

    selected_train_files = [
        "gs://data2moe/train/part-00-of-161.jsonl",
        "gs://data2moe/train/part-05-of-161.jsonl",
        "gs://data2moe/train/part-06-of-161.jsonl",
        "gs://data2moe/train/part-07-of-161.jsonl",
        "gs://data2moe/train/part-08-of-161.jsonl",
        "gs://data2moe/train/part-09-of-161.jsonl",
        "gs://data2moe/train/part-10-of-161.jsonl",
        "gs://data2moe/train/part-104-of-161.jsonl",
        "gs://data2moe/train/part-107-of-161.jsonl",
        "gs://data2moe/train/part-108-of-161.jsonl",
    ]
    patch_pattern = "gs://data2moe/image_patch/part-*.jsonl.gz"

    options = PipelineOptions(beam_args, save_main_session=True, runtime_type_check=True)

  
        # 读取 train
    with beam.Pipeline(options=options) as p:
        train_pc = (
            p
            | "TrainFiles" >> beam.Create(selected_train_files)    # 列表 → PCollection[str]
            | "ReadTrain"  >> beam.io.ReadAllFromText()            # 批量读取列表里的文件
            | "TrainKV"    >> beam.Map(train2kv)
            | "FilterTrainKV" >> beam.Filter(lambda kv: kv and kv[0])
        )

        patch_pc = (
            p
            | "PatchGlob" >> beam.Create([patch_pattern])
            | "ReadPatch" >> beam.io.ReadAllFromText(
                                compression_type=CompressionTypes.GZIP
                            )
            | "PatchKV"   >> beam.Map(patch2kv)
            | "FilterPatchKV" >> beam.Filter(lambda kv: kv and kv[0])
        )
        merged = (
            {"train": train_pc, "patch": patch_pc}
            | "CoGroupByParent" >> beam.CoGroupByKey()
            | "MergePatch" >> beam.FlatMap(merge_patch)
        )
        (
            merged
            | "Batch" >> beam.BatchElements(min_batch_size=1024, max_batch_size=1024)
            | "WriteWDS" >> beam.ParDo(WriteWebDataset(args.output_dir, compress=True))
        )

if __name__ == "__main__":
    run()
