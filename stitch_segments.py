#!/usr/bin/env python3
"""把目录里的分段视频按文件名顺序拼成一个连续视频，便于直接观看跨段连续性。

独立小工具，不依赖项目里的推理代码，也不改动任何入口。

音轨处理（重要）：
    分段 mp4 的音轨是 AAC，每段的末尾都会被补齐到 1024 采样的帧边界，因此**每段都带约
    30 ms 的近静音尾巴**（例：89 帧 = 4.45 s = 71200 采样，文件里却是 71680 采样）。
    如果把各段解码后的音轨直接首尾相接，每个拼接点就会多出 30 ms 静音 —— 听起来是
    "音频跳变/中断"，而且音轨相对画面会逐段累积漂移。

    因此本工具默认：
      * `--audio-source` 给了就用源音频重建整条音轨（最干净：一次编码、采样边界精确）；
      * 没给则按"视频时长 × 采样率"裁掉每段 AAC 补零（`--trim-aac-padding`，默认开）；
      * 音轨比视频短时用静音补齐到视频长度（`--pad-audio-to-video`，默认开）。

用法：
    # 推荐：用 audio/ 里的源音频做音轨
    python stitch_segments.py output/segmented_89f_fixed --audio-source audio

    # 或者用一个长音频文件（按分段顺序依次切片）
    python stitch_segments.py output/segmented_89f_fixed --audio-source cn_test_0.wav

    # 不用源音频（回落为"裁剪 AAC 补零"）
    python stitch_segments.py output/segmented_89f_fixed

    # 每段第一帧冻结 20 帧，方便逐帧对比边界
    python stitch_segments.py output/segmented_89f_fixed --audio-source audio --pause-frames 20

默认会在每帧左上角叠加 "segNN f### g####"，并在每段第一帧画红框；同时打印（并写出）
每段的起止全局帧号与音轨裁剪情况，便于定位段边界。
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def collect_files(directory: Path, pattern: str | None) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"目录不存在：{directory}")
    files = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if pattern:
        files = [p for p in files if fnmatch.fnmatch(p.name, pattern)]
    # 跳过本工具此前的输出，避免把上一次的拼接结果当成输入（会缺音轨/自覆盖）
    kept = [p for p in files if not p.name.startswith("stitched_")]
    skipped = [p.name for p in files if p.name.startswith("stitched_")]
    if skipped:
        print(f"跳过 {len(skipped)} 个本工具的历史输出：{skipped}")
    return kept


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """解码整条音轨为 (channels, samples) float32（统一到 [-1, 1]，与 AAC 解码结果同尺度）。"""
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise SystemExit(f"没有音轨：{path}")
        sample_rate = stream.codec_context.sample_rate
        parts = [frame.to_ndarray() for frame in container.decode(audio=0)]
    if not parts:
        raise SystemExit(f"音轨为空：{path}")
    audio = np.concatenate(parts, axis=1)
    if audio.dtype.kind in "iu":          # PCM 源（如 wav 的 int16）需要归一到 [-1, 1]
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    return audio, sample_rate


def collect_source_audio(source: Path | None) -> tuple[Path | None, list[Path]]:
    """返回 (单个长音频文件, 音频目录里的文件列表)。"""
    if source is None:
        return None, []
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
        if not files:
            raise SystemExit(f"目录里没有音频文件：{source}")
        return None, files
    if source.is_file():
        return source, []
    raise SystemExit(f"--audio-source 既不是文件也不是目录：{source}")


def match_source_audio(segment: Path, audio_files: list[Path], index: int) -> Path | None:
    """优先按文件名前缀匹配（段名以音频名开头），否则按下标顺序兜底。"""
    for candidate in audio_files:
        if segment.name.startswith(candidate.stem):
            return candidate
    return audio_files[index] if index < len(audio_files) else None


def annotate(
    rgb: np.ndarray,
    *,
    segment: int,
    frame_in_segment: int,
    global_frame: int,
    boundary: bool,
    labels: bool,
    mark: bool,
) -> np.ndarray:
    if not labels and not (mark and boundary):
        return rgb
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if mark and boundary:
        draw.rectangle([0, 0, image.width - 1, image.height - 1], outline=(255, 0, 0), width=2)
    if labels:
        draw.text((4, 3), f"seg{segment:02d}  f{frame_in_segment:03d}  g{global_frame:04d}", fill=(255, 255, 0))
    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="包含分段 mp4 的目录（按文件名排序拼接）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 mp4（默认 <目录>/stitched_<目录名>.mp4）")
    parser.add_argument("--pattern", default=None, help="只拼接匹配该通配符的文件，例如 'cn_test_0_seg0*.mp4'")
    parser.add_argument(
        "--audio-source",
        type=Path,
        default=None,
        help="音轨来源（推荐）：音频目录（按文件名与分段匹配）或一个长音频文件（按分段顺序切片）",
    )
    parser.add_argument(
        "--trim-aac-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="没有 --audio-source 时，按 视频时长×采样率 裁掉每段 AAC 帧对齐补零（默认开）",
    )
    parser.add_argument(
        "--pad-audio-to-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="音轨总长不足视频时用静音补齐到视频长度（默认开）",
    )
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True, help="叠加段号/帧号角标（默认开）")
    parser.add_argument(
        "--boundary-mark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在每段第一帧画红框，便于看到段边界（默认开）",
    )
    parser.add_argument(
        "--pause-frames",
        type=int,
        default=0,
        help="每段第一帧重复 N 帧做冻结（便于逐帧对比边界；会让画面比音频稍慢，仅用于检视）",
    )
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--preset", default="fast")
    args = parser.parse_args()

    out_path = args.output or (args.directory / f"stitched_{args.directory.name}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 输出文件就在输入目录里时，必须把它自己排除掉，否则第二次运行会自覆盖。
    files = [p for p in collect_files(args.directory, args.pattern) if p.resolve() != out_path.resolve()]
    if not files:
        raise SystemExit(f"目录里没有可拼接的视频：{args.directory}")

    single_source, source_dir_files = collect_source_audio(args.audio_source)

    with av.open(str(files[0])) as probe:
        v0 = next(s for s in probe.streams if s.type == "video")
        a0 = next((s for s in probe.streams if s.type == "audio"), None)
        width, height, fps = v0.width, v0.height, float(v0.average_rate)
        audio_rate = a0.codec_context.sample_rate if a0 is not None else None
        audio_layout = a0.codec_context.layout.name if a0 is not None else None
        audio_format = a0.codec_context.format.name if a0 is not None else None

    print(f"输入：{args.directory}（{len(files)} 个视频）")
    print(f"输出：{out_path}")
    print(f"视频：{width}x{height} @ {fps:g} fps；音频：{audio_rate} Hz {audio_layout} {audio_format}")
    if single_source is not None:
        print(f"音轨来源：长音频 {single_source}（按分段顺序切片）")
    elif source_dir_files:
        print(f"音轨来源：音频目录 {args.audio_source}（{len(source_dir_files)} 个文件，按名称匹配分段）")
    else:
        print(f"音轨来源：分段自身（trim_aac_padding={args.trim_aac_padding}）")

    container = av.open(str(out_path), mode="w")
    vstream = container.add_stream("libx264", rate=int(round(fps)))
    vstream.width, vstream.height, vstream.pix_fmt = width, height, "yuv420p"
    vstream.options = {"crf": str(args.crf), "preset": args.preset}
    astream = None
    if audio_rate:
        astream = container.add_stream("aac", rate=audio_rate)
        astream.layout = audio_layout
        astream.format = audio_format

    # 单文件源音频只解码一次，之后按分段顺序切片
    source_audio: np.ndarray | None = None
    if single_source is not None:
        source_audio, source_rate = decode_audio(single_source)
        if source_rate != audio_rate:
            raise SystemExit(f"源音频采样率 {source_rate} 与视频音轨 {audio_rate} 不一致")
        if source_audio.shape[0] != 2:
            source_audio = np.repeat(source_audio.mean(axis=0, keepdims=True), 2, axis=0).astype(np.float32)
        print(f"源音频：{single_source} {source_audio.shape[1]} 采样（{source_audio.shape[1]/source_rate:.3f}s）")

    rows: list[dict] = []
    audio_parts: list[np.ndarray] = []
    global_frame = 0
    source_cursor = 0  # 单文件源音频的累计采样位置

    for segment, path in enumerate(files):
        frame_count = 0
        with av.open(str(path)) as source:
            for index, frame in enumerate(source.decode(video=0)):
                rgb = frame.to_ndarray(format="rgb24")
                boundary = segment > 0 and index == 0
                rgb = annotate(
                    rgb,
                    segment=segment,
                    frame_in_segment=index,
                    global_frame=global_frame + index,
                    boundary=boundary,
                    labels=args.labels,
                    mark=args.boundary_mark,
                )
                if boundary and args.pause_frames > 0:
                    for _ in range(args.pause_frames):
                        frame_out = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        for packet in vstream.encode(frame_out):
                            container.mux(packet)
                frame_out = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in vstream.encode(frame_out):
                    container.mux(packet)
                frame_count += 1

        info = {"segment": segment, "name": path.name, "frames": frame_count, "start": global_frame}
        global_frame += frame_count
        info["end"] = global_frame - 1

        # ---------------- 音轨 ----------------
        if astream is not None:
            expected = int(round(frame_count / fps * audio_rate))
            trimmed = 0
            padded = 0
            if source_audio is not None:
                piece = source_audio[:, source_cursor : source_cursor + expected]
                source_cursor += expected
                info["audio_source"] = single_source.name
            else:
                if source_dir_files:
                    chosen = match_source_audio(path, source_dir_files, segment)
                    if chosen is None:
                        raise SystemExit(f"找不到分段 {path.name} 对应的源音频")
                    block, src_rate = decode_audio(chosen)
                    if src_rate != audio_rate:
                        raise SystemExit(f"源音频 {chosen} 采样率 {src_rate} != {audio_rate}")
                    if block.shape[0] != 2:
                        block = np.repeat(block.mean(axis=0, keepdims=True), 2, axis=0)
                    info["audio_source"] = chosen.name
                else:
                    block, src_rate = decode_audio(path)
                    if src_rate != audio_rate:
                        raise SystemExit(f"分段 {path.name} 音轨采样率 {src_rate} != {audio_rate}")
                if args.trim_aac_padding and block.shape[1] > expected:
                    trimmed = block.shape[1] - expected
                    piece = block[:, :expected]
                else:
                    piece = block
            if piece.shape[1] < expected:
                padded = expected - piece.shape[1]
                piece = np.concatenate([piece, np.zeros((piece.shape[0], padded), dtype=piece.dtype)], axis=1)
            audio_parts.append(piece)
            info.update({"audio_expected": expected, "audio_used": piece.shape[1],
                         "audio_trimmed": trimmed, "audio_padded": padded})

        rows.append(info)
        extra = ""
        if "audio_trimmed" in info:
            extra = f"  音轨 应={info['audio_expected']} 实={info['audio_used']} 裁={info['audio_trimmed']} 补={info['audio_padded']}"
        print(f"  seg{segment:02d}: {path.name}  {frame_count} 帧  全局 {info['start']}–{info['end']}{extra}")

    for packet in vstream.encode():
        container.mux(packet)

    audio_seconds = 0.0
    if astream is not None and audio_parts:
        audio = np.concatenate(audio_parts, axis=1)
        video_samples = int(round(global_frame / fps * audio_rate))
        if args.pad_audio_to_video and audio.shape[1] < video_samples:
            pad = video_samples - audio.shape[1]
            audio = np.concatenate([audio, np.zeros((audio.shape[0], pad), dtype=audio.dtype)], axis=1)
            print(f"音轨尾部补静音 {pad} 采样（{pad/audio_rate*1000:.0f} ms）以对齐视频长度")
        step = 1024  # AAC 帧长
        for start in range(0, audio.shape[1], step):
            block = audio[:, start : start + step]
            if block.shape[1] < step:
                block = np.concatenate(
                    [block, np.zeros((block.shape[0], step - block.shape[1]), dtype=block.dtype)], axis=1
                )
            audio_frame = av.AudioFrame.from_ndarray(block, format=audio_format, layout=audio_layout)
            audio_frame.sample_rate = audio_rate
            audio_frame.pts = start
            audio_frame.time_base = Fraction(1, audio_rate)
            for packet in astream.encode(audio_frame):
                container.mux(packet)
        for packet in astream.encode(None):
            container.mux(packet)
        audio_seconds = audio.shape[1] / audio_rate

    container.close()

    video_seconds = (global_frame + max(0, len(files) - 1) * args.pause_frames) / fps
    print(f"\n总帧数 {global_frame}（约 {video_seconds:.2f}s），音轨 {audio_seconds:.2f}s -> {out_path}")
    print("段边界（全局帧号）：")
    for row in rows[1:]:
        print(f"  seg{row['segment']:02d} 起始于全局帧 {row['start']}（段内第 1 帧，红框标注）")

    report = out_path.with_suffix(".segments.txt")
    with report.open("w", encoding="utf-8") as handle:
        handle.write(f"# source_dir: {args.directory}\n")
        handle.write(f"# output: {out_path}\n")
        handle.write(f"# fps {fps:g}, pause_frames {args.pause_frames}, audio_source {args.audio_source}\n")
        handle.write("segment\tfile\tframes\tglobal_start\tglobal_end\taudio_expected\taudio_used\taudio_trimmed\taudio_padded\n")
        for row in rows:
            handle.write(
                f"{row['segment']}\t{row['name']}\t{row['frames']}\t{row['start']}\t{row['end']}\t"
                f"{row.get('audio_expected', '')}\t{row.get('audio_used', '')}\t"
                f"{row.get('audio_trimmed', '')}\t{row.get('audio_padded', '')}\n"
            )
    print(f"边界与音轨清单：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
