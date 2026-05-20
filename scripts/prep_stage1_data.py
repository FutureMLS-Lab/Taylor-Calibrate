"""
Quick script to rechunk existing all_tokens.npy to seqlen=512 for Stage 1.
Avoids re-downloading by symlinking the existing token cache.

Usage:
    python prep_stage1_data.py
"""
import os
import numpy as np
import pyarrow as pa
from datasets import Dataset

SRC_NPY = os.environ.get("SRC_NPY", "data/all_tokens.npy")
DST_DIR = os.environ.get("DST_DIR", "data/stage1")
DST_NPY = os.path.join(DST_DIR, "all_tokens.npy")
CTX = 512

def main():
    os.makedirs(DST_DIR, exist_ok=True)

    if not os.path.exists(DST_NPY):
        print(f"Symlinking {SRC_NPY} -> {DST_NPY}")
        os.symlink(SRC_NPY, DST_NPY)

    out_path = os.path.join(DST_DIR, f"chunked_context{CTX}")
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}, skipping.")
        return

    print(f"Loading tokens from {DST_NPY}...")
    all_tokens = np.load(DST_NPY, mmap_mode='r')
    total_len = (len(all_tokens) // CTX) * CTX
    chunks = np.array(all_tokens[:total_len]).reshape(-1, CTX)
    print(f"Total tokens: {total_len:,}, Chunks: {len(chunks):,} x {CTX}")

    arrow_type = pa.uint32()
    flat_arr = pa.array(chunks.flatten(), type=arrow_type)
    arrow_arr = pa.FixedSizeListArray.from_arrays(flat_arr, CTX)
    table = pa.Table.from_arrays([arrow_arr], names=["input_ids"])
    chunked_ds = Dataset(table)

    print(f"Saving to {out_path}...")
    chunked_ds.save_to_disk(out_path)
    print(f"Done! {len(chunks):,} chunks of length {CTX}")


if __name__ == "__main__":
    main()
