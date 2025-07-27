
"""

从 Hugging Face 以 streaming 模式加载 Amazon Reviews 2023 的
Sports_and_Outdoors 子集，transform 字段并分片压缩（.jsonl.gz）后
直接写入 Google Cloud Storage，供 Beam pipeline 并行读取。

- 支持 `--prefix` 参数，自定义 GCS 路径前缀（默认 sports_and_outdoors）
- 对 reviews 和 meta 都调用 with_format(type="python")，绕过 PyArrow 的 cast
- 分片大小可通过参数调整
"""

import os
import json
import gzip
import argparse

from datasets import load_dataset
from huggingface_hub import login
from google.cloud import storage


def transform_review(ex):
    return {
        "user_id":        ex.get("user_id"),
        "asin":           ex.get("asin"),
        "parent_asin":    ex.get("parent_asin", ex.get("asin")),
        "rating":         ex.get("rating"),
        "title":          ex.get("title", ""),
        "text":           ex.get("text", ""),
        "sort_timestamp": ex.get("timestamp"),
        "verified_purchase": ex.get("verified_purchase", False),
        "helpful_votes":  ex.get("helpful_vote", 0),
    }




def upload_jsonl_gzip_shards(dataset, bucket_name, blob_prefix, transform_fn, shard_size=300_000):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    writer = None
    gz = None
    for idx, example in enumerate(dataset):
        if idx % shard_size == 0:
            if gz:
                gz.close()
                writer.close()
            shard_id = idx // shard_size
            shard_path = f"{blob_prefix}-shard-{shard_id:05d}.jsonl.gz"
            blob = bucket.blob(shard_path)
            writer = blob.open("wb")
            gz = gzip.GzipFile(fileobj=writer, mode="w")
            print(f"→ open new shard: gs://{bucket_name}/{shard_path}")

        record = transform_fn(example)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        gz.write(line.encode("utf-8"))

        if (idx + 1) % 100_000 == 0:
            print(f"  written {idx + 1} records…")

    if gz:
        gz.close()
        writer.close()
        total_shards = (idx // shard_size) + 1
        print(f"All done. {idx + 1} records in {total_shards} shards.")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming upload Amazon-Reviews to GCS"
    )
    parser.add_argument("--bucket",            type=str, required=True,
                        help="GCS bucket name (e.g. my-bucket)")
    parser.add_argument("--prefix",            type=str, default="sports_and_outdoors",
                        help="GCS path prefix inside the bucket")
    parser.add_argument("--review_shard_size", type=int, default=300_000,
                        help="每个 review shard 的记录数")

    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("请先设置环境变量 HF_TOKEN")
    login(token=hf_token)

    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        name="raw_review_Sports_and_Outdoors",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    # 使用 with_format
    reviews = reviews.with_format(type="python")



    # 上传 Reviews
    upload_jsonl_gzip_shards(
        dataset=reviews,
        bucket_name=args.bucket,
        blob_prefix=f"{args.prefix}/reviews",
        transform_fn=transform_review,
        shard_size=args.review_shard_size,
    )


if __name__ == "__main__":
    main()
