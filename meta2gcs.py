
import os
import json
import gzip
import argparse
import requests
from google.cloud import storage
from huggingface_hub import hf_hub_url

def stream_meta_via_http(repo_id: str, split_name: str, token: str):
    # 拼出原始文件的 raw URL
    url = hf_hub_url(
        repo_id=repo_id,
        filename=f"raw/meta_categories/meta_{split_name}.jsonl",
        repo_type="dataset",
    )
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    # 按行迭代，不会一次性下载整个文件
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            yield json.loads(line)

def transform_meta(ex: dict) -> dict:
    # 兼容旧 schema(dict-of-lists) 和新 schema(list-of-structs)
    raw = ex.get("images", {}) or []
    imgs = []
    if isinstance(raw, dict):
        for var, hi, lg, th in zip(raw.get("variant", []),
                                  raw.get("hi_res", []),
                                  raw.get("large", []),
                                  raw.get("thumb", [])):
            imgs.append({"variant":var,"hi_res":hi,"large":lg,"thumb":th})
    else:
        for it in raw:
            imgs.append({
                "variant": it.get("variant"),
                "hi_res":   it.get("hi_res"),
                "large":    it.get("large"),
                "thumb":    it.get("thumb"),
            })
    # 清洗 price
    price = None
    rp = ex.get("price")
    if rp not in (None, "", "None"):
        try: price = float(rp)
        except: price = None
    # 解析 details 字符串
    details = ex.get("details", {}) or {}
    if isinstance(details, str):
        try: details = json.loads(details)
        except: details = {}

    return {
        "parent_asin":     ex.get("parent_asin"),
        "asin":            ex.get("asin"),
        "main_category":   ex.get("main_category"),
        "title":           ex.get("title"),
        "average_rating":  ex.get("average_rating"),
        "rating_number":   ex.get("rating_number"),
        "price":           price,
        "store":           ex.get("store"),
        "features":        ex.get("features", []),
        "description":     ex.get("description", []),
        "details":         details,
        "images":          imgs,
        "bought_together": ex.get("bought_together"),
        "categories":      ex.get("categories"),
    }


def upload_jsonl_gzip_shards(iterator, bucket_name, prefix, shard_size):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    gz = None
    writer = None

    for idx, rec in enumerate(iterator):
        if idx % shard_size == 0:
            if gz:
                gz.close(); writer.close()
            sid = idx // shard_size
            filename = f"shard-{sid:05d}.jsonl.gz"
            # 直接在 prefix 子目录下写入
            blob_path = f"{prefix}/{filename}"
            blob = bucket.blob(blob_path)
            writer = blob.open("wb")
            gz = gzip.GzipFile(fileobj=writer, mode="w")
            print(f"→ open new shard: gs://{bucket_name}/{blob_path}")

        out = transform_meta(rec)
        gz.write((json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8"))
        if (idx + 1) % 50000 == 0:
            print(f"  written {idx+1} records…")

    if gz:
        gz.close(); writer.close()
        total = idx + 1
        shards = (idx // shard_size) + 1
        print(f"All done. {total} records in {shards} shards.")

        print(f"All done. {total} records in {shards} shards.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",     required=True, help="GCS bucket 名称")
    p.add_argument("--prefix",     default="meta", help="GCS 路径前缀")
    p.add_argument("--shard_size", type=int, default=100000, help="分片大小")
    p.add_argument("--split",      default="Sports_and_Outdoors",
                   help="如 Sports_and_Outdoors, All_Beauty 等")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("请先设置 HF_TOKEN 环境变量")

    iterator = stream_meta_via_http(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        split_name=args.split,
        token=hf_token
    )
    upload_jsonl_gzip_shards(
        iterator=iterator,
        bucket_name=args.bucket,
        prefix=args.prefix,
        shard_size=args.shard_size
    )

if __name__ == "__main__":
    main()
