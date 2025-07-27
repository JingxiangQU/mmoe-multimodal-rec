#!/usr/bin/env python3
import io
import json
import base64
import logging
import argparse
import asyncio
import httpx
import base64
import numpy as np
import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.options.pipeline_options import (
    PipelineOptions, StandardOptions, SetupOptions
)
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor
# 每条 process() 批量并发下载的 URL 数量


class MyOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            "--input_pattern", type=str,
            help="GCS glob for input JSONL shards, e.g. gs://bucket/image_urls/part-*-of-30.jsonl"
        )
        parser.add_value_provider_argument(
            "--output_prefix", type=str,
            help="GCS prefix for output, e.g. gs://bucket/image_patch/part"
        )
        parser.add_value_provider_argument(
            "--num_shards", type=int, default=30,
            help="Number of output shards"
        )

PATCH = 16                # patch size 16x16
MAX_BATCH_SIZE = 64       # 根据显存和吞吐适当调整
MAX_CONCURRENT = 8        # http并发限制

class BatchAsyncImageToPatchDoFn(beam.DoFn):
    def setup(self):
        self.Image = PILImage
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

    def process(self, batch):
        if len(batch) > MAX_BATCH_SIZE:
            logging.warning(f"[batch warning] batch_size={len(batch)}过大，自动切分")
            for i in range(0, len(batch), MAX_BATCH_SIZE):
                yield from self.process(batch[i:i+MAX_BATCH_SIZE])
            return

        urls = [rec["url"] for rec in batch]
        pids = [rec["parent_asin"] for rec in batch]

        async def fetch(idx, url, client):
            try:
                r = await client.get(url)
                r.raise_for_status()
                return idx, r.content
            except Exception as e:
                logging.warning(f"[download failed] {url}: {e}")
                return idx, None

        async def download_all(urls):
            limits = httpx.Limits(max_connections=MAX_CONCURRENT)
            async with httpx.AsyncClient(timeout=10, limits=limits) as client:
                tasks = [fetch(i, u, client) for i, u in enumerate(urls)]
                return await asyncio.gather(*tasks, return_exceptions=False)

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(download_all(urls))
        loop.close()

        work_items = []
        for idx, content in results:
            if content:
                work_items.append((idx, content, pids[idx]))

        def decode_and_patch(item):
            idx, content, pid = item
            try:
                img = self.Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                logging.warning(f"[invalid image] {urls[idx]}: {e}")
                return None

            img = img.resize((256, 256), self.Image.BILINEAR)
            left = (256 - 224) // 2
            img = img.crop((left, left, left + 224, left + 224))
            arr = np.asarray(img, dtype=np.uint8)
            if arr.shape != (224, 224, 3):
                logging.warning(f"[shape error] {urls[idx]} shape={arr.shape}, 跳过")
                return None
            arr = arr.transpose(2, 0, 1)  # CHW
            C, H, W = arr.shape
            p = PATCH
            if H % p != 0 or W % p != 0:
                logging.warning(f"[patch error] {urls[idx]} 不能整除PATCH, shape={arr.shape}")
                return None

            patches_np = arr.reshape(C, H//p, p, W//p, p) \
                            .transpose(1, 3, 0, 2, 4) \
                            .reshape(-1, C * p * p)  # (196,768)
            b64 = base64.b64encode(patches_np.tobytes()).decode()
            return {
                "parent_asin": pid,
                "patch_b64": b64,
                "shape": [patches_np.shape[0], C, p, p]
            }

        for out in self.executor.map(decode_and_patch, work_items):
            if out is not None:
                yield out

    def teardown(self):
        self.executor.shutdown(wait=True)


def run(argv=None):
    parser = argparse.ArgumentParser()
    # 上面 ValueProvider 不用 argparse
    args, beam_args = parser.parse_known_args(argv)

    options = PipelineOptions(beam_args)
    options.view_as(SetupOptions).save_main_session = True
    std = options.view_as(StandardOptions)
    logging.info(f">>> Using runner = {std.runner}")

    opts = options.view_as(MyOptions)
    # 从 ValueProvider 里拿真实值
    try:
        input_pattern = opts.input_pattern.get()
        output_prefix = opts.output_prefix.get()
        num_shards    = opts.num_shards   .get()
    except Exception:
        input_pattern = opts.input_pattern
        output_prefix = opts.output_prefix
        num_shards    = opts.num_shards

    with beam.Pipeline(options=options) as p:
        (p
         # 1) 逐行读取 JSONL
         | "ReadJSONL"    >> ReadFromText(input_pattern)
         | "ParseJson"    >> beam.Map(json.loads)
         # 2) 批量聚集成 lists，每批 BATCH_SIZE 条
         | "BatchElements" >> beam.BatchElements(
               min_batch_size=MAX_BATCH_SIZE,
               max_batch_size=MAX_BATCH_SIZE)
         # 3) 并发下载 + 切 patch
         | "BatchAsyncPatch" >> beam.ParDo(BatchAsyncImageToPatchDoFn())
         # 4) JSON 序列化 + 写回 GCS
         | "FilterValidPatch" >> beam.Filter(lambda r: r is not None)
         | "ToJson"       >> beam.Map(lambda r: json.dumps(r, ensure_ascii=False))
         | "WriteOut"     >> WriteToText(
               f"{output_prefix.rstrip('/')}/image_patch/part",
               file_name_suffix     = ".jsonl.gz",
               num_shards           = num_shards,
               shard_name_template  = "-SS-of-NN",
               compression_type     = "gzip" )
        )

if __name__ == "__main__":
    run()
