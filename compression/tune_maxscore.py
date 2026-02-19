#!/usr/bin/env python3
"""
Parameter sweep utility for max-score commaVQ compression.
"""

from __future__ import annotations

import argparse
import itertools
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
        encode_segment_to_words,
        load_gpt_model,
        save_json,
    )
except ImportError:
    from compression.maxscore_core import (
        FRAMES_PER_SEGMENT,
        TOKENS_PER_FRAME,
        MaxScoreConfig,
        build_temporal_bias_lookup,
        challenge_data_files,
        encode_segment_to_words,
        load_gpt_model,
        save_json,
    )


def ensure_constriction():
    try:
        import constriction

        return constriction
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "constriction~=0.4.2"])
        import constriction

        return constriction


def parse_float_grid(text: str) -> list[float]:
    values = [v.strip() for v in text.split(",")]
    parsed = [float(v) for v in values if v]
    if not parsed:
        raise ValueError("Grid cannot be empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune max-score compression parameters")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-segments", type=int, default=25, help="Number of leading segments to score each config")
    parser.add_argument("--precision-bits", type=int, default=16)
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Optional model URL/path passed to utils.gpt loader (e.g. file:///.../pytorch_model.bin)",
    )
    parser.add_argument("--temperature-grid", type=str, default="1.0,0.95")
    parser.add_argument("--temporal-boost-grid", type=str, default="2.5,3.0")
    parser.add_argument("--temporal-sigma-grid", type=str, default="20,30")
    parser.add_argument("--copy-bonus-grid", type=str, default="1.5,2.0")
    parser.add_argument("--max-combos", type=int, default=0, help="0 means all combinations")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path(__file__).resolve().parent / "maxscore_best_config.json",
        help="Best config output JSON (consumed by compress_maxscore.py)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).resolve().parent / "maxscore_tuning_report.json",
        help="Full sweep metrics report",
    )
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constriction = ensure_constriction()
    model_family = constriction.stream.model.Categorical(perfect=False)

    temp_grid = parse_float_grid(args.temperature_grid)
    boost_grid = parse_float_grid(args.temporal_boost_grid)
    sigma_grid = parse_float_grid(args.temporal_sigma_grid)
    copy_grid = parse_float_grid(args.copy_bonus_grid)

    combos = list(itertools.product(temp_grid, boost_grid, sigma_grid, copy_grid))
    if args.max_combos > 0:
        combos = combos[: args.max_combos]
    if not combos:
        raise ValueError("No parameter combinations to evaluate")

    from datasets import load_dataset

    print("Loading challenge dataset subset...")
    ds = load_dataset("commaai/commavq", data_files=challenge_data_files(), split="train")
    if args.num_segments <= 0:
        raise ValueError("--num-segments must be positive")
    if args.num_segments > len(ds):
        raise ValueError(f"--num-segments={args.num_segments} exceeds available {len(ds)}")

    num_segments = int(args.num_segments)
    total_tokens = num_segments * FRAMES_PER_SEGMENT * TOKENS_PER_FRAME

    model_url = args.model_url
    if not model_url:
        local_model = (Path(__file__).resolve().parent.parent / "gpt2m" / "pytorch_model.bin").resolve()
        if local_model.exists():
            model_url = local_model.as_uri()
    print(f"Loading GPT model on {args.device} ...")
    if model_url:
        print(f"Model source: {model_url}")
        model, _ = load_gpt_model(device=args.device, model_url=model_url)
    else:
        model, _ = load_gpt_model(device=args.device)
    device = torch.device(args.device)

    best_cfg: MaxScoreConfig | None = None
    best_rate = -1.0
    best_bpt = float("inf")
    results: list[dict[str, float]] = []

    search_start = time.time()
    for combo_idx, (temperature, temporal_boost, temporal_sigma, copy_bonus) in enumerate(combos):
        cfg = MaxScoreConfig(
            precision_bits=args.precision_bits,
            temperature=temperature,
            temporal_boost=temporal_boost,
            temporal_sigma=temporal_sigma,
            copy_bonus=copy_bonus,
        )
        cfg.validate()
        bias_lookup = build_temporal_bias_lookup(cfg)

        cfg_start = time.time()
        total_words = 0
        for seg_idx in range(num_segments):
            tokens = np.asarray(ds[seg_idx]["token.npy"], dtype=np.int16)
            words = encode_segment_to_words(
                model=model,
                device=device,
                segment_tokens=tokens,
                config=cfg,
                model_family=model_family,
                bias_lookup=bias_lookup,
            )
            total_words += int(words.size)

        bits_per_token = (total_words * 32.0) / float(total_tokens)
        rate = 10.0 / bits_per_token
        elapsed = time.time() - cfg_start

        result = {
            "temperature": float(temperature),
            "temporal_boost": float(temporal_boost),
            "temporal_sigma": float(temporal_sigma),
            "copy_bonus": float(copy_bonus),
            "bits_per_token": float(bits_per_token),
            "estimated_rate": float(rate),
            "elapsed_sec": float(elapsed),
        }
        results.append(result)

        if rate > best_rate:
            best_rate = rate
            best_bpt = bits_per_token
            best_cfg = cfg

        if (combo_idx + 1) % args.log_every == 0 or combo_idx == len(combos) - 1:
            print(
                f"[{combo_idx+1}/{len(combos)}] temp={temperature:.3f} boost={temporal_boost:.3f} "
                f"sigma={temporal_sigma:.3f} copy={copy_bonus:.3f} -> "
                f"{rate:.4f}x ({bits_per_token:.4f} bpt), {elapsed:.1f}s"
            )

    if best_cfg is None:
        raise RuntimeError("No successful tuning result")

    search_elapsed = time.time() - search_start
    best_payload = best_cfg.to_dict()
    save_json(args.output_config, best_payload)
    save_json(
        args.output_report,
        {
            "num_segments": num_segments,
            "total_combos": len(combos),
            "best": best_payload,
            "best_estimated_rate": best_rate,
            "best_bits_per_token": best_bpt,
            "elapsed_sec": search_elapsed,
            "results": sorted(results, key=lambda item: item["estimated_rate"], reverse=True),
        },
    )

    print("")
    print("Tuning complete.")
    print(f"Best config: {best_payload}")
    print(f"Best estimated rate: {best_rate:.4f}x")
    print(f"Best bits/token: {best_bpt:.4f}")
    print(f"Best config JSON: {args.output_config}")
    print(f"Report JSON: {args.output_report}")
    print(f"Total elapsed: {search_elapsed/3600.0:.2f} hours")


if __name__ == "__main__":
    main()
