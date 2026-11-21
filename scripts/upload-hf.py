#!/usr/bin/env python3
"""Caphlon-2 Hugging Face Publisher.

Uploads the standalone 4.52 GiB PTQ1_SB GGUF, metadata, kernel patch, and model card
to Hugging Face (`ssuasd/Caphlon-2` or user-specified `--repo-id`).

Usage:
  HF_TOKEN=hf_... python3 scripts/upload-hf.py \
    --gguf /Users/demir/Desktop/theleaf/colab/calisma/cand_stage2_merged_standalone.gguf \
    --repo-id ssuasd/Caphlon-2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload Caphlon-2 artifacts to Hugging Face Hub")
    ap.add_argument("--repo-id", default="ssuasd/Caphlon-2", help="Target HF repo (e.g. ssuasd/Caphlon-2)")
    ap.add_argument(
        "--gguf",
        type=Path,
        default=Path("/Users/demir/Desktop/theleaf/colab/calisma/cand_stage2_merged_standalone.gguf"),
        help="Path to local 4.52 GiB PTQ1_SB standalone GGUF",
    )
    ap.add_argument(
        "--target-gguf-name",
        default="Caphlon-2-27B-PTQ1_SB.gguf",
        help="Filename in the Hugging Face repository",
    )
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face write token (or set HF_TOKEN)")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: HF_TOKEN is not set. Export HF_TOKEN=hf_... or pass --token hf_...", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    api = HfApi(token=args.token)
    user = api.whoami()
    print(f"[HF] Authenticated as: {user.get('name')}")

    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"[HF] Repo ready: https://huggingface.co/{args.repo_id}")

    # 1. Upload support files first (fast)
    support_files = [
        (repo_root / "README.md", "README.md"),
        (repo_root / "NOTICE.md", "NOTICE.md"),
        (repo_root / "LICENSE", "LICENSE"),
        (repo_root / "CITATION.cff", "CITATION.cff"),
        (repo_root / "patches/prismll_ptq1_sb_cuda_metal.patch", "patches/prismll_ptq1_sb_cuda_metal.patch"),
        (repo_root / "metadata/Caphlon-2-27B-PTQ1_SB.meta.json", "metadata/Caphlon-2-27B-PTQ1_SB.meta.json"),
        (repo_root / "scripts/build-prism-llamacpp.sh", "scripts/build-prism-llamacpp.sh"),
        (repo_root / "scripts/run.sh", "scripts/run.sh"),
        (repo_root / "scripts/eval-en40.sh", "scripts/eval-en40.sh"),
    ]
    for src, dst in support_files:
        if src.exists():
            print(f"[HF] Uploading {dst} ({src.stat().st_size} B)...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dst,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {dst} for Caphlon-2 (4.52 GiB PTQ1_SB)",
            )

    # 2. Upload the 4.52 GiB standalone GGUF
    if args.gguf.exists():
        sz = args.gguf.stat().st_size
        print(f"[HF] Uploading {args.gguf} -> {args.target_gguf_name} ({sz:,} bytes / {sz / 2**30:.4f} GiB)...")
        api.upload_file(
            path_or_fileobj=str(args.gguf),
            path_in_repo=args.target_gguf_name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload {args.target_gguf_name} (4.52 GiB PTQ1_SB standalone GGUF)",
        )
        print(f"[HF] DONE: https://huggingface.co/{args.repo_id}")
    else:
        print(f"WARNING: GGUF file not found at {args.gguf}, skipped weight upload.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
