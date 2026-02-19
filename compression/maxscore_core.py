#!/usr/bin/env python3
"""
Shared core logic for max-score commaVQ compression.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

MAGIC = b"CVQMAX1\x00"
HEADER_STRUCT = struct.Struct("<8sIIffff")

TOKENS_PER_FRAME = 128
FRAMES_PER_SEGMENT = 1200
VOCAB_SIZE = 1024
BOS_TOKEN = 1024

MAX_SEQUENCE_TOKENS = 20 * 129
MAX_CONTEXT_TOKENS = MAX_SEQUENCE_TOKENS - (TOKENS_PER_FRAME + 1)

MODEL_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin"
PROB_EPS = 1e-9


@dataclass(frozen=True)
class MaxScoreConfig:
    precision_bits: int = 16
    temperature: float = 1.0
    temporal_boost: float = 3.0
    temporal_sigma: float = 30.0
    copy_bonus: float = 2.0

    def validate(self) -> None:
        if self.precision_bits <= 0:
            raise ValueError("precision_bits must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.temporal_sigma <= 0.0:
            raise ValueError("temporal_sigma must be > 0")
        if self.temporal_boost <= 0.0:
            raise ValueError("temporal_boost must be > 0")
        if self.copy_bonus <= 0.0:
            raise ValueError("copy_bonus must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaxScoreConfig":
        allowed = {k: data[k] for k in cls.__dataclass_fields__.keys() if k in data}  # type: ignore[attr-defined]
        cfg = cls(**allowed)
        cfg.validate()
        return cfg


def challenge_data_files() -> dict[str, list[str]]:
    return {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]}


def raw_bytes_for_segments(num_segments: int) -> int:
    return num_segments * FRAMES_PER_SEGMENT * TOKENS_PER_FRAME * 10 // 8


def compression_rate(num_segments: int, archive_size_bytes: int) -> float:
    if archive_size_bytes <= 0:
        raise ValueError("archive_size_bytes must be positive")
    return raw_bytes_for_segments(num_segments) / archive_size_bytes


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config_json(path: Path, fallback: MaxScoreConfig | None = None) -> MaxScoreConfig:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "best" in data and isinstance(data["best"], dict):
        data = data["best"]
    if fallback is not None:
        merged = fallback.to_dict()
        if isinstance(data, dict):
            for key in merged:
                if key in data:
                    merged[key] = data[key]
        return MaxScoreConfig.from_dict(merged)
    if not isinstance(data, dict):
        raise ValueError(f"Config JSON must be an object: {path}")
    return MaxScoreConfig.from_dict(data)


def _import_gpt():
    try:
        from utils.gpt import GPT, GPTConfig

        return GPT, GPTConfig
    except ImportError:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from utils.gpt import GPT, GPTConfig

        return GPT, GPTConfig


def load_gpt_model(device: str, model_url: str = MODEL_URL):
    GPT, GPTConfig = _import_gpt()
    gpt_config = GPTConfig()
    with torch.device("meta"):
        model = GPT(gpt_config)
    try:
        model.load_state_dict_from_url(model_url, assign=True)
    except TypeError:
        model.load_state_dict_from_url(model_url)
    model = model.eval().to(device=device, dtype=torch.float32)
    for param in model.parameters():
        param.requires_grad_(False)
    # Keep KV cache dtype aligned with model activations to avoid index_put dtype mismatch on CPU.
    model.setup_caches(max_batch_size=1, max_seq_length=gpt_config.block_size)
    for block in model.transformer.h:
        kv = block.attn.kv_cache
        if kv is None:
            continue
        if kv.k_cache.dtype != torch.float32:
            kv.k_cache = kv.k_cache.float()
        if kv.v_cache.dtype != torch.float32:
            kv.v_cache = kv.v_cache.float()
    return model, gpt_config


def prepare_segment_tokens(tokens: np.ndarray) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.int16)
    if arr.shape == (FRAMES_PER_SEGMENT, 8, 16):
        arr = arr.reshape(FRAMES_PER_SEGMENT, TOKENS_PER_FRAME)
    elif arr.shape != (FRAMES_PER_SEGMENT, TOKENS_PER_FRAME):
        raise ValueError(f"Unexpected token shape: {arr.shape}")
    out = arr.astype(np.int32, copy=False)
    min_token = int(out.min())
    max_token = int(out.max())
    if min_token < 0 or max_token >= VOCAB_SIZE:
        raise ValueError(f"Token range out of bounds: [{min_token}, {max_token}]")
    return out


def reshape_segment_for_output(tokens_2d: np.ndarray) -> np.ndarray:
    return np.asarray(tokens_2d, dtype=np.int16).reshape(FRAMES_PER_SEGMENT, 8, 16)


def trim_context_tokens(context_tokens: Sequence[int]) -> list[int]:
    if len(context_tokens) <= MAX_CONTEXT_TOKENS:
        return list(context_tokens)
    trimmed = list(context_tokens[-MAX_CONTEXT_TOKENS:])
    if trimmed and trimmed[0] != BOS_TOKEN:
        try:
            first_bos = trimmed.index(BOS_TOKEN)
            trimmed = trimmed[first_bos:]
        except ValueError:
            trimmed = []
    return trimmed


def build_temporal_bias_lookup(config: MaxScoreConfig) -> np.ndarray:
    ids = np.arange(VOCAB_SIZE, dtype=np.float64)
    distances = np.abs(ids[None, :] - ids[:, None])
    sigma = max(config.temporal_sigma, 1e-6)
    lookup = 1.0 + (config.temporal_boost - 1.0) * np.exp(-0.5 * (distances / sigma) ** 2)
    diag = np.arange(VOCAB_SIZE)
    lookup[diag, diag] *= config.copy_bonus
    return lookup


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    out = np.asarray(probs, dtype=np.float64)
    out = np.maximum(out, PROB_EPS)
    out /= out.sum(dtype=np.float64)
    return out.astype(np.float32)


def apply_temporal_bias(
    probs: np.ndarray,
    prev_token: int,
    bias_lookup: np.ndarray,
) -> np.ndarray:
    biased = np.asarray(probs, dtype=np.float64) * bias_lookup[int(prev_token)]
    return _normalize_probs(biased)


def _logits_to_probs(logits: torch.Tensor, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return _normalize_probs(probs.detach().cpu().numpy())


@torch.no_grad()
def _prefill_next_logits(
    model: torch.nn.Module,
    device: torch.device,
    prefix_tokens: list[int],
) -> tuple[torch.Tensor, int]:
    prefix_np = np.asarray(prefix_tokens, dtype=np.int64)
    if prefix_np.ndim != 1 or prefix_np.size == 0:
        raise ValueError("prefix_tokens must be a non-empty 1D sequence")
    token_tensor = torch.from_numpy(prefix_np).to(device=device, dtype=torch.long).unsqueeze(0)
    input_pos = torch.arange(prefix_np.size, device=device, dtype=torch.long)
    logits = model(token_tensor, input_pos)
    next_logits = logits[0, -1, :VOCAB_SIZE]
    return next_logits, int(prefix_np.size)


@torch.no_grad()
def compute_frame_probability_matrix(
    model: torch.nn.Module,
    device: torch.device,
    context_tokens: list[int],
    frame_tokens: np.ndarray,
    config: MaxScoreConfig,
    bias_lookup: np.ndarray,
    prev_frame_tokens: np.ndarray | None,
) -> np.ndarray:
    if frame_tokens.shape != (TOKENS_PER_FRAME,):
        raise ValueError(f"Expected frame shape ({TOKENS_PER_FRAME},), got {frame_tokens.shape}")

    prefix_tokens = list(context_tokens) + [BOS_TOKEN]
    next_logits, prefix_len = _prefill_next_logits(model, device, prefix_tokens)

    token_step = torch.empty((1, 1), device=device, dtype=torch.long)
    pos_step = torch.empty((1,), device=device, dtype=torch.long)
    probs_matrix = np.empty((TOKENS_PER_FRAME, VOCAB_SIZE), dtype=np.float32)

    for pos in range(TOKENS_PER_FRAME):
        probs = _logits_to_probs(next_logits, config.temperature)
        if prev_frame_tokens is not None:
            probs = apply_temporal_bias(probs, int(prev_frame_tokens[pos]), bias_lookup)
        probs_matrix[pos] = probs

        if pos == TOKENS_PER_FRAME - 1:
            break

        token_step[0, 0] = int(frame_tokens[pos])
        pos_step[0] = prefix_len + pos
        step_logits = model(token_step, pos_step)
        next_logits = step_logits[0, 0, :VOCAB_SIZE]

    return probs_matrix


@torch.no_grad()
def decode_frame_tokens(
    model: torch.nn.Module,
    device: torch.device,
    context_tokens: list[int],
    config: MaxScoreConfig,
    bias_lookup: np.ndarray,
    prev_frame_tokens: np.ndarray | None,
    decoder: Any,
    model_family: Any,
) -> np.ndarray:
    prefix_tokens = list(context_tokens) + [BOS_TOKEN]
    next_logits, prefix_len = _prefill_next_logits(model, device, prefix_tokens)

    token_step = torch.empty((1, 1), device=device, dtype=torch.long)
    pos_step = torch.empty((1,), device=device, dtype=torch.long)
    decoded = np.empty((TOKENS_PER_FRAME,), dtype=np.int32)

    for pos in range(TOKENS_PER_FRAME):
        probs = _logits_to_probs(next_logits, config.temperature)
        if prev_frame_tokens is not None:
            probs = apply_temporal_bias(probs, int(prev_frame_tokens[pos]), bias_lookup)

        symbol = int(decoder.decode(model_family, probs[np.newaxis, :])[0])
        decoded[pos] = symbol

        if pos == TOKENS_PER_FRAME - 1:
            break

        token_step[0, 0] = symbol
        pos_step[0] = prefix_len + pos
        step_logits = model(token_step, pos_step)
        next_logits = step_logits[0, 0, :VOCAB_SIZE]

    return decoded


@torch.no_grad()
def encode_segment_to_words(
    model: torch.nn.Module,
    device: torch.device,
    segment_tokens: np.ndarray,
    config: MaxScoreConfig,
    model_family: Any,
    bias_lookup: np.ndarray,
) -> np.ndarray:
    import constriction

    tokens = prepare_segment_tokens(segment_tokens)
    encoder = constriction.stream.queue.RangeEncoder()

    context_tokens: list[int] = []
    prev_frame_tokens: np.ndarray | None = None

    for frame_idx in range(FRAMES_PER_SEGMENT):
        context_tokens = trim_context_tokens(context_tokens)
        frame = tokens[frame_idx]
        probs = compute_frame_probability_matrix(
            model=model,
            device=device,
            context_tokens=context_tokens,
            frame_tokens=frame,
            config=config,
            bias_lookup=bias_lookup,
            prev_frame_tokens=prev_frame_tokens,
        )
        encoder.encode(frame, model_family, probs)
        context_tokens.append(BOS_TOKEN)
        context_tokens.extend(frame.tolist())
        prev_frame_tokens = frame

    return np.array(encoder.get_compressed(), dtype=np.uint32, copy=True)


@torch.no_grad()
def decode_segment_from_words(
    model: torch.nn.Module,
    device: torch.device,
    words: np.ndarray,
    config: MaxScoreConfig,
    model_family: Any,
    bias_lookup: np.ndarray,
) -> np.ndarray:
    import constriction

    decoder = constriction.stream.queue.RangeDecoder(np.asarray(words, dtype=np.uint32))
    decoded = np.empty((FRAMES_PER_SEGMENT, TOKENS_PER_FRAME), dtype=np.int16)

    context_tokens: list[int] = []
    prev_frame_tokens: np.ndarray | None = None

    for frame_idx in range(FRAMES_PER_SEGMENT):
        context_tokens = trim_context_tokens(context_tokens)
        frame = decode_frame_tokens(
            model=model,
            device=device,
            context_tokens=context_tokens,
            config=config,
            bias_lookup=bias_lookup,
            prev_frame_tokens=prev_frame_tokens,
            decoder=decoder,
            model_family=model_family,
        )
        decoded[frame_idx] = frame.astype(np.int16)
        context_tokens.append(BOS_TOKEN)
        context_tokens.extend(frame.tolist())
        prev_frame_tokens = frame

    return decoded


def write_data_bin(
    data_bin_path: Path,
    config: MaxScoreConfig,
    segment_word_lengths: np.ndarray,
    segment_paths: Sequence[Path],
) -> None:
    config.validate()
    lengths = np.asarray(segment_word_lengths, dtype=np.uint32)
    if lengths.ndim != 1:
        raise ValueError("segment_word_lengths must be rank-1")
    if len(segment_paths) != int(lengths.size):
        raise ValueError("segment path count must match segment_word_lengths")

    data_bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_bin_path, "wb") as out:
        out.write(
            HEADER_STRUCT.pack(
                MAGIC,
                int(lengths.size),
                int(config.precision_bits),
                float(config.temperature),
                float(config.temporal_boost),
                float(config.temporal_sigma),
                float(config.copy_bonus),
            )
        )
        out.write(lengths.tobytes())
        for path in segment_paths:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, out, length=1 << 20)


def read_data_bin_header(data_bin_path: Path) -> tuple[MaxScoreConfig, np.ndarray, int]:
    with open(data_bin_path, "rb") as f:
        raw = f.read(HEADER_STRUCT.size)
        if len(raw) != HEADER_STRUCT.size:
            raise ValueError("data.bin is too small to contain a header")
        magic, num_segments, precision_bits, temperature, temporal_boost, temporal_sigma, copy_bonus = HEADER_STRUCT.unpack(
            raw
        )
        if magic != MAGIC:
            raise ValueError(f"Unexpected data.bin magic: {magic!r}")
        lengths = np.fromfile(f, dtype=np.uint32, count=int(num_segments))
        if lengths.size != int(num_segments):
            raise ValueError("data.bin ended while reading segment lengths")
        offset = HEADER_STRUCT.size + int(num_segments) * 4

    config = MaxScoreConfig(
        precision_bits=int(precision_bits),
        temperature=float(temperature),
        temporal_boost=float(temporal_boost),
        temporal_sigma=float(temporal_sigma),
        copy_bonus=float(copy_bonus),
    )
    config.validate()
    return config, lengths, offset


def iter_words_from_data_bin(
    data_bin_path: Path,
    segment_word_lengths: np.ndarray,
    payload_offset: int,
):
    with open(data_bin_path, "rb") as f:
        f.seek(payload_offset)
        for length in np.asarray(segment_word_lengths, dtype=np.uint32):
            n_words = int(length)
            n_bytes = n_words * 4
            raw = f.read(n_bytes)
            if len(raw) != n_bytes:
                raise ValueError("data.bin ended while reading a segment payload")
            yield np.frombuffer(raw, dtype=np.uint32).copy()
