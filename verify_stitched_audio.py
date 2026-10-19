#!/usr/bin/env python3
"""检查拼接视频的音轨在每个拼接点是否与源音频对齐、是否存在静音缺口。

对每个段边界做三件事：
  1. 在拼接音轨里以"该边界应有位置"为中心搜索与源音频最匹配的偏移 → 对齐误差（采样/ms）；
  2. 按该偏移对齐后计算残差 RMS（相对源音频 RMS 的比例）；
  3. 统计边界附近最长的一段近静音（|x| < 1e-3）长度 → 直接暴露"AAC 补零被接进成品"的问题。

用法：
    python verify_stitched_audio.py output/tp1_segmented_noglobalsink/stitched_tp1_segmented_noglobalsink.mp4 \
        --segments output/tp1_segmented_noglobalsink/stitched_tp1_segmented_noglobalsink.segments.txt \
        --source cn_test_0.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np


def decode_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise SystemExit(f"没有音轨：{path}")
        rate = stream.codec_context.sample_rate
        parts = [frame.to_ndarray() for frame in container.decode(audio=0)]
    audio = np.concatenate(parts, axis=1)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    return audio.mean(axis=0), rate


def parse_segments(path: Path) -> list[int]:
    frames: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("segment\t"):
            continue
        parts = line.split("\t")
        frames.append(int(parts[2]))
    return frames


def best_lag(stitched: np.ndarray, source: np.ndarray, *, expected: int, window: int, probe: int) -> tuple[int, float, float]:
    """在 expected±window 内找与源音频最匹配的位置，返回 (lag, 残差RMS, 源RMS)。"""
    ref_start = max(0, expected - probe // 2)
    ref = source[ref_start : ref_start + probe]
    if ref.size < probe:
        ref = source[-probe:]
        ref_start = source.size - probe
    best = (0, float("inf"))
    for lag in range(-window, window + 1):
        start = ref_start + lag
        if start < 0 or start + ref.size > stitched.size:
            continue
        cand = stitched[start : start + ref.size]
        if cand.size != ref.size:
            continue
        err = float(np.sqrt(np.mean((cand - ref) ** 2)))
        if err < best[1]:
            best = (lag, err)
    return best[0], best[1], float(np.sqrt(np.mean(ref**2)))


def longest_silence(x: np.ndarray, threshold: float = 1e-3) -> int:
    mask = np.abs(x) < threshold
    if not mask.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    runs = idx[1::2] - idx[0::2]
    return int(runs.max()) if runs.size else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stitched", type=Path)
    parser.add_argument("--segments", type=Path, required=True, help="stitch_segments.py 写出的 *.segments.txt")
    parser.add_argument("--source", type=Path, required=True, help="源音频（拼接各段对应的原始音频）")
    parser.add_argument("--fps", type=float, default=None, help="默认从拼接视频读取")
    parser.add_argument("--search-window", type=int, default=8000, help="边界搜索范围（采样）")
    parser.add_argument("--probe", type=int, default=4000, help="用于比对的窗口长度（采样）")
    args = parser.parse_args()

    frames = parse_segments(args.segments)
    stitched, sr = decode_audio_mono(args.stitched)
    source, src_rate = decode_audio_mono(args.source)
    if src_rate != sr:
        raise SystemExit(f"采样率不一致：拼接 {sr} vs 源 {src_rate}")
    fps = args.fps
    if fps is None:
        with av.open(str(args.stitched)) as container:
            video = next(s for s in container.streams if s.type == "video")
            fps = float(video.average_rate)

    print(f"拼接音轨 {stitched.size} 采样（{stitched.size/sr:.3f}s）；源音频 {source.size} 采样（{source.size/sr:.3f}s）；"
          f"{len(frames)} 段 @ {fps:g} fps")
    print(f"\n{'边界':>4} {'应有采样':>10} {'实测偏移':>10} {'等于':>9} {'残差RMS':>10} {'相对源RMS':>10} {'边界附近最长静音':>16}")

    cursor = 0
    lags, silence = [], []
    for index in range(len(frames) - 1):
        cursor += int(round(frames[index] / fps * sr))
        lag, resid, ref_rms = best_lag(
            stitched, source, expected=cursor, window=args.search_window, probe=args.probe
        )
        near = stitched[max(0, cursor + lag - 1500) : cursor + lag + 1500]
        gap = longest_silence(near)
        lags.append(lag)
        silence.append(gap)
        print(f"{index+1:>4} {cursor:>10d} {lag:>+10d} {lag/sr*1000:>+8.1f}ms {resid:>10.4f} "
              f"{resid/max(ref_rms, 1e-9):>10.2%} {gap:>16d}")

    print(f"\n汇总：对齐误差 |lag| 平均 {np.mean(np.abs(lags)):.1f} 采样（{np.mean(np.abs(lags))/sr*1000:.1f} ms）、"
          f"最大 {np.max(np.abs(lags))} 采样；边界附近最长静音 平均 {np.mean(silence):.0f}、最大 {np.max(silence)} 采样"
          f"（{np.max(silence)/sr*1000:.0f} ms）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
