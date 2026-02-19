#!/usr/bin/env python3
"""
Max-score commaVQ compressor.

Builds `compression_challenge_submission.zip` containing:
- data.bin
- decompress.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from maxscore_core import (
        FRAMES_PER_SEGMENT,
        TOKENS_PER_FRAME,
        MaxScoreConfig,
        build_temporal_bias_lookup,
        challenge_data_files,
        compression_rate,
        decode_segment_from_words,
        encode_segment_to_words,
        load_config_json,
        load_gpt_model,
        raw_bytes_for_segments,
        save_json,
        write_data_bin,
    )
except ImportError:
    from compression.maxscore_core import (
        FRAMES_PER_SEGMENT,
        TOKENS_PER_FRAME,
        MaxScoreConfig,
        build_temporal_bias_lookup,
        challenge_data_files,
        compression_rate,
        decode_segment_from_words,
        encode_segment_to_words,
        load_config_json,
        load_gpt_model,
        raw_bytes_for_segments,
        save_json,
        write_data_bin,
    )


HERE = Path(__file__).resolve().parent


def ensure_constriction():
    try:
        import constriction

        return constriction
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "constriction~=0.4.2"])
        import constriction

        return constriction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Max-score commaVQ compressor")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-segments", type=int, default=5000, help="Number of segments from the start of split")
    parser.add_argument("--precision-bits", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temporal-boost", type=float, default=3.0)
    parser.add_argument("--temporal-sigma", type=float, default=30.0)
    parser.add_argument("--copy-bonus", type=float, default=2.0)
    parser.add_argument("--config-json", type=Path, default=None, help="JSON file from tune_maxscore.py")
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Optional model URL/path passed to utils.gpt loader (e.g. file:///.../pytorch_model.bin)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=HERE / "maxscore_work",
        help="Directory for resumable segment payloads and staging",
    )
    parser.add_argument(
        "--submission-zip",
        type=Path,
        default=HERE / "compression_challenge_submission.zip",
        help="Final output zip",
    )
    parser.add_argument("--force-recompress", action="store_true", help="Ignore existing resumable segment payloads")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--self-check-segment", action="store_true", help="Decode one segment after compression sanity-check")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MaxScoreConfig:
    cfg = MaxScoreConfig(
        precision_bits=args.precision_bits,
        temperature=args.temperature,
        temporal_boost=args.temporal_boost,
        temporal_sigma=args.temporal_sigma,
        copy_bonus=args.copy_bonus,
    )
    if args.config_json is not None:
        cfg = load_config_json(args.config_json, fallback=cfg)
    cfg.validate()
    return cfg


def write_segment_words(path: Path, words: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(np.asarray(words, dtype=np.uint32).tobytes())
    os.replace(tmp, path)


def maybe_self_check_segment(
    model: torch.nn.Module,
    device: torch.device,
    cfg: MaxScoreConfig,
    model_family,
    bias_lookup: np.ndarray,
    tokens: np.ndarray,
    words: np.ndarray,
) -> None:
    decoded = decode_segment_from_words(
        model=model,
        device=device,
        words=words,
        config=cfg,
        model_family=model_family,
        bias_lookup=bias_lookup,
    )
    gt = np.asarray(tokens, dtype=np.int16).reshape(FRAMES_PER_SEGMENT, TOKENS_PER_FRAME)
    if not np.array_equal(decoded, gt):
        raise AssertionError("Self-check failed: decoded segment differs from source tokens")


def stage_submission(stage_dir: Path, data_bin_path: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_bin_path, stage_dir / "data.bin")
    shutil.copy2(HERE / "decompress_maxscore.py", stage_dir / "decompress.py")


def make_submission_zip(stage_dir: Path, zip_path: Path) -> Path:
    target_zip = zip_path if zip_path.suffix == ".zip" else Path(str(zip_path) + ".zip")
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    if target_zip.exists():
        target_zip.unlink()
    base_name = str(target_zip)[:-4]
    created = Path(shutil.make_archive(base_name, "zip", root_dir=stage_dir))
    if created.resolve() != target_zip.resolve():
        if target_zip.exists():
            target_zip.unlink()
        os.replace(created, target_zip)
    return target_zip


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    constriction = ensure_constriction()
    model_family = constriction.stream.model.Categorical(perfect=False)

    from datasets import load_dataset

    print("Loading challenge dataset split (data-0000 + data-0001)...")
    ds = load_dataset("commaai/commavq", data_files=challenge_data_files(), split="train")
    if args.num_segments > len(ds):
        raise ValueError(f"--num-segments={args.num_segments} exceeds available {len(ds)}")

    num_segments = int(args.num_segments)
    if num_segments != 5000:
        print(f"WARNING: using {num_segments} segments (full challenge uses 5000)")

    model_url = args.model_url
    if not model_url:
        local_model = (HERE.parent / "gpt2m" / "pytorch_model.bin").resolve()
        if local_model.exists():
            model_url = local_model.as_uri()
    print(f"Loading GPT model on {args.device} ...")
    if model_url:
        print(f"Model source: {model_url}")
        model, _ = load_gpt_model(device=args.device, model_url=model_url)
    else:
        model, _ = load_gpt_model(device=args.device)
    device = torch.device(args.device)
    bias_lookup = build_temporal_bias_lookup(cfg)

    work_dir: Path = args.work_dir
    segments_dir = work_dir / "segments"
    data_bin_path = work_dir / "data.bin"
    stage_dir = work_dir / "submission_stage"
    segments_dir.mkdir(parents=True, exist_ok=True)

    save_json(work_dir / "used_config.json", cfg.to_dict())

    segment_lengths = np.zeros((num_segments,), dtype=np.uint32)
    total_words = 0
    t_start = time.time()

    for idx in range(num_segments):
        seg_path = segments_dir / f"{idx:05d}.u32"
        if seg_path.exists() and not args.force_recompress:
            n_words = seg_path.stat().st_size // 4
            segment_lengths[idx] = n_words
            total_words += n_words
            if (idx + 1) % args.log_every == 0 or idx == num_segments - 1:
                elapsed = time.time() - t_start
                print(f"[{idx+1}/{num_segments}] resume {seg_path.name} ({n_words} words), elapsed={elapsed:.1f}s")
            continue

        example = ds[idx]
        file_name = example["json"]["file_name"]
        tokens = np.asarray(example["token.npy"], dtype=np.int16)

        words = encode_segment_to_words(
            model=model,
            device=device,
            segment_tokens=tokens,
            config=cfg,
            model_family=model_family,
            bias_lookup=bias_lookup,
        )
        write_segment_words(seg_path, words)

        if args.self_check_segment and idx == 0:
            print("Running self-check decode on first segment...")
            maybe_self_check_segment(
                model=model,
                device=device,
                cfg=cfg,
                model_family=model_family,
                bias_lookup=bias_lookup,
                tokens=tokens,
                words=words,
            )

        n_words = int(words.size)
        segment_lengths[idx] = n_words
        total_words += n_words

        if (idx + 1) % args.log_every == 0 or idx == num_segments - 1:
            elapsed = time.time() - t_start
            est_rate_no_zip = (10.0 * (idx + 1) * FRAMES_PER_SEGMENT * TOKENS_PER_FRAME) / (total_words * 32.0)
            print(
                f"[{idx+1}/{num_segments}] {file_name} -> {n_words} words, "
                f"running(data-only)~{est_rate_no_zip:.3f}x, elapsed={elapsed:.1f}s"
            )

    segment_paths = [segments_dir / f"{i:05d}.u32" for i in range(num_segments)]
    for path in segment_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing segment payload: {path}")

    print("Writing data.bin ...")
    write_data_bin(
        data_bin_path=data_bin_path,
        config=cfg,
        segment_word_lengths=segment_lengths,
        segment_paths=segment_paths,
    )

    print("Staging submission files ...")
    stage_submission(stage_dir=stage_dir, data_bin_path=data_bin_path)
    zip_path = make_submission_zip(stage_dir=stage_dir, zip_path=args.submission_zip.resolve())

    zip_size = zip_path.stat().st_size
    data_bin_size = data_bin_path.stat().st_size
    rate = compression_rate(num_segments=num_segments, archive_size_bytes=zip_size)
    raw_bytes = raw_bytes_for_segments(num_segments)
    total_elapsed = time.time() - t_start

    print("")
    print("Done.")
    print(f"Segments:                {num_segments}")
    print(f"Raw bytes (10-bit ref):  {raw_bytes}")
    print(f"data.bin bytes:          {data_bin_size}")
    print(f"zip bytes:               {zip_size}")
    print(f"Estimated rate:          {rate:.4f}x")
    print(f"Output zip:              {zip_path}")
    print(f"Total time:              {total_elapsed/3600.0:.2f} hours")


if __name__ == "__main__":
    main()
