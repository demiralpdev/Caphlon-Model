#!/usr/bin/env python3
"""Caphlon-2 Hugging Face Publisher (Cand_pure / cand_tam_bsqrt — 48/60 = 80.0% GPQA-D, 236/270 = 87.4% Halu-270).

Uploads the winning 4.52 GiB PTQ1_SB GGUF (`Orig_PTQ1_0_cand_tam_bsqrt_15456_SB.gguf`),
ITI steering vector (`iti_mean.gguf`), V83K metadata, kernel patch, and model card
to Hugging Face (`ssuasd/Caphlon-2` or user-specified `--repo-id`).

If the local `Caphlon-2-27B-PTQ1_SB.gguf` is not yet in `colab/calisma/`, it automatically
fetches it from `gdrive:theleaf-arsiv/colab/calisma/gecici/Orig_PTQ1_0_cand_tam_bsqrt_15456_SB.gguf`
via `rclone` and verifies its SHA-256 (`ac41fcddf5da28d71198f60202c10ea37cc64d791e5f6f118f849a02579c20db`).

Usage:
  HF_TOKEN=hf_... python3 scripts/upload-hf.py --repo-id ssuasd/Caphlon-2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_BYTES = 4856213824
EXPECTED_SHA256 = "ac41fcddf5da28d71198f60202c10ea37cc64d791e5f6f118f849a02579c20db"
GDRIVE_SRC = "gdrive:theleaf-arsiv/colab/calisma/gecici/Orig_PTQ1_0_cand_tam_bsqrt_15456_SB.gguf"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ensure_cand_pure_gguf(gguf_path: Path) -> Path:
    if not gguf_path.exists() or gguf_path.stat().st_size != EXPECTED_BYTES:
        print(f"[RCLONE] Fetching Cand_pure (cand_tam_bsqrt) from {GDRIVE_SRC} -> {gguf_path}...")
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rclone", "copyto", "--progress", GDRIVE_SRC, str(gguf_path)],
            check=True,
        )
    sz = gguf_path.stat().st_size
    if sz != EXPECTED_BYTES:
        raise RuntimeError(f"Size mismatch for {gguf_path}: got {sz}, expected {EXPECTED_BYTES}")
    print(f"[VERIFY] Computing SHA-256 of {gguf_path} ({sz:,} B)...")
    digest = sha256_file(gguf_path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {gguf_path}!\n"
            f"  got:      {digest}\n"
            f"  expected: {EXPECTED_SHA256} (Cand_pure / cand_tam_bsqrt 48/60 GPQA-D)"
        )
    print(f"[VERIFY] SHA-256 OK: {digest}")
    return gguf_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload Caphlon-2 (Cand_pure 48/60 GPQA-D) to Hugging Face Hub")
    ap.add_argument("--repo-id", default="ssuasd/Caphlon-2", help="Target HF repo (e.g. ssuasd/Caphlon-2)")
    ap.add_argument(
        "--gguf",
        type=Path,
        default=Path("/Users/demir/Desktop/theleaf/colab/calisma/Caphlon-2-27B-PTQ1_SB.gguf"),
        help="Path to local 4.52 GiB Cand_pure (cand_tam_bsqrt) PTQ1_SB GGUF",
    )
    ap.add_argument(
        "--target-gguf-name",
        default="Caphlon-2-27B-PTQ1_SB.gguf",
        help="Filename in the Hugging Face repository",
    )
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or None, help="Hugging Face write token (defaults to cached hf auth login token)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    gguf_file = ensure_cand_pure_gguf(args.gguf)

    api = HfApi(token=args.token)
    user = api.whoami()
    print(f"[HF] Authenticated as: {user.get('name')}")

    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"[HF] Repo ready: https://huggingface.co/{args.repo_id}")

    support_files = [
        (repo_root / "NOTICE.md", "NOTICE.md"),
        (repo_root / "LICENSE", "LICENSE"),
        (repo_root / "CITATION.cff", "CITATION.cff"),
        (repo_root / "patches/prismll_ptq1_sb_cuda_metal.patch", "patches/prismll_ptq1_sb_cuda_metal.patch"),
        (repo_root / "metadata/Caphlon-2-27B-PTQ1_SB.meta.json", "metadata/Caphlon-2-27B-PTQ1_SB.meta.json"),
        (repo_root / "metadata/iti_mean.gguf", "metadata/iti_mean.gguf"),
        (repo_root / "scripts/build-prism-llamacpp.sh", "scripts/build-prism-llamacpp.sh"),
        (repo_root / "scripts/run.sh", "scripts/run.sh"),
        (repo_root / "scripts/eval-en40.sh", "scripts/eval-en40.sh"),
    ]
    readme_src = repo_root / "README.md"
    if readme_src.exists():
        raw_readme = readme_src.read_text(encoding="utf-8")
        if not raw_readme.startswith("---\n"):
            hf_frontmatter = (
                "---\n"
                "license: apache-2.0\n"
                "base_model:\n"
                "  - Qwen/Qwen3.8-27B\n"
                "  - prism-ml/Ternary-Bonsai-2-27B-gguf\n"
                "  - ukisai/Swift-Bonsai-2-GGUF\n"
                "library_name: gguf\n"
                "pipeline_tag: text-generation\n"
                "language:\n"
                "  - en\n"
                "  - tr\n"
                "tags:\n"
                "  - gguf\n"
                "  - caphlon\n"
                "  - caphlon-2\n"
                "  - qwen3.8\n"
                "  - bonsai\n"
                "  - ternary\n"
                "  - ptq1_sb\n"
                "  - llama.cpp\n"
                "---\n\n"
            )
            raw_readme = hf_frontmatter + raw_readme
        print(f"[HF] Uploading README.md ({len(raw_readme)} B)...")
        api.upload_file(
            path_or_fileobj=raw_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update README.md for Caphlon-2 (4.52 GiB PTQ1_SB, 80.0% GPQA-D)",
        )
    for src, dst in support_files:
        if src.exists():
            print(f"[HF] Uploading {dst} ({src.stat().st_size} B)...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dst,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {dst} for Caphlon-2 (4.52 GiB PTQ1_SB, 80.0% GPQA-D)",
            )

    sz = gguf_file.stat().st_size
    print(f"[HF] Uploading {gguf_file} -> {args.target_gguf_name} ({sz:,} bytes / {sz / 2**30:.4f} GiB)...")
    api.upload_file(
        path_or_fileobj=str(gguf_file),
        path_in_repo=args.target_gguf_name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {args.target_gguf_name} (Cand_pure 4.52 GiB PTQ1_SB, SHA256 ac41fcdd...)",
    )
    print(f"[HF] DONE: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
