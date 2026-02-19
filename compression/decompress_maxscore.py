#!/usr/bin/env python3
"""
Max-score commaVQ decompressor.

This file is copied into the final submission zip as `decompress.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def ensure_constriction():
    try:
        import constriction

        return constriction
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "constriction~=0.4.2"])
        import constriction

        return constriction


def import_core_symbols():
    try:
        from maxscore_core import (
            FRAMES_PER_SEGMENT,
            MaxScoreConfig,
            build_temporal_bias_lookup,
            challenge_data_files,
            decode_segment_from_words,
            iter_words_from_data_bin,
            load_gpt_model,
            read_data_bin_header,
            reshape_segment_for_output,
        )

        return (
            FRAMES_PER_SEGMENT,
            MaxScoreConfig,
            build_temporal_bias_lookup,
            challenge_data_files,
            decode_segment_from_words,
            iter_words_from_data_bin,
            load_gpt_model,
            read_data_bin_header,
            reshape_segment_for_output,
        )
    except ImportError:
        here = Path(__file__).resolve()
        repo_root_candidates = [
            here.parent.parent.parent,  # repo root when running from compression/<unzipped>/decompress.py
            here.parent.parent,  # repo root when running from compression/decompress_maxscore.py
            here.parent,
        ]
        for root in repo_root_candidates:
            if (root / "compression" / "maxscore_core.py").exists():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                break
        from compression.maxscore_core import (
            FRAMES_PER_SEGMENT,
            MaxScoreConfig,
            build_temporal_bias_lookup,
            challenge_data_files,
            decode_segment_from_words,
            iter_words_from_data_bin,
            load_gpt_model,
            read_data_bin_header,
            reshape_segment_for_output,
        )

        return (
            FRAMES_PER_SEGMENT,
            MaxScoreConfig,
            build_temporal_bias_lookup,
            challenge_data_files,
            decode_segment_from_words,
            iter_words_from_data_bin,
            load_gpt_model,
            read_data_bin_header,
            reshape_segment_for_output,
        )


(
    FRAMES_PER_SEGMENT,
    MaxScoreConfig,
    build_temporal_bias_lookup,
    challenge_data_files,
    decode_segment_from_words,
    iter_words_from_data_bin,
    load_gpt_model,
    read_data_bin_header,
    reshape_segment_for_output,
) = import_core_symbols()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Max-score commaVQ decompressor")
    parser.add_argument(
        "--data-bin",
        type=Path,
        default=Path(__file__).resolve().parent / "data.bin",
        help="Path to data.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parent / "decompressed")),
        help="Directory to write decompressed files",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Optional model URL/path passed to utils.gpt loader (e.g. file:///.../pytorch_model.bin)",
    )
    parser.add_argument("--num-segments", type=int, default=None, help="Decode only first N segments")
    parser.add_argument("--verify", action="store_true", help="Verify each decoded segment against dataset tokens")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constriction = ensure_constriction()
    model_family = constriction.stream.model.Categorical(perfect=False)

    data_bin_path: Path = args.data_bin
    output_dir: Path = args.output_dir
    if not data_bin_path.exists():
        raise FileNotFoundError(f"Missing data.bin: {data_bin_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config, segment_lengths, payload_offset = read_data_bin_header(data_bin_path)
    config.validate()
    total_segments = int(segment_lengths.size)
    num_segments = total_segments if args.num_segments is None else min(int(args.num_segments), total_segments)

    from datasets import load_dataset

    print("Loading challenge dataset split for filenames...")
    ds = load_dataset("commaai/commavq", data_files=challenge_data_files(), split="train")
    if num_segments > len(ds):
        raise ValueError(f"data.bin has {num_segments} segments but dataset has only {len(ds)}")

    model_url = args.model_url
    if not model_url:
        here = Path(__file__).resolve()
        candidate_roots = [here.parent, here.parent.parent, here.parent.parent.parent]
        for root in candidate_roots:
            local_model = (root / "gpt2m" / "pytorch_model.bin").resolve()
            if local_model.exists():
                model_url = local_model.as_uri()
                break
    print(f"Loading GPT model on {args.device} ...")
    if model_url:
        print(f"Model source: {model_url}")
        model, _ = load_gpt_model(device=args.device, model_url=model_url)
    else:
        model, _ = load_gpt_model(device=args.device)
    device = torch.device(args.device)
    bias_lookup = build_temporal_bias_lookup(config)

    print(f"Decoding {num_segments}/{total_segments} segments...")
    words_iter = iter_words_from_data_bin(data_bin_path, segment_lengths[:num_segments], payload_offset)
    for idx, words in enumerate(words_iter):
        decoded_2d = decode_segment_from_words(
            model=model,
            device=device,
            words=words,
            config=config,
            model_family=model_family,
            bias_lookup=bias_lookup,
        )
        decoded = reshape_segment_for_output(decoded_2d)

        if args.verify:
            gt = np.asarray(ds[idx]["token.npy"], dtype=np.int16)
            if not np.array_equal(decoded, gt):
                raise AssertionError(f"Decoded tokens mismatch at segment index {idx}")

        file_name = ds[idx]["json"]["file_name"]
        out_path = output_dir / file_name
        np.save(out_path, decoded)
        # np.save appends ".npy"; evaluator expects bare file name.
        os.replace(str(out_path) + ".npy", out_path)

        if (idx + 1) % args.log_every == 0 or idx == num_segments - 1:
            print(f"[{idx+1}/{num_segments}] {file_name}")

    print("Done.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
