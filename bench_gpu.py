"""Sample GPU utilization while running a short inference (for parallelism feasibility analysis)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

NUM_FRAMES = "241"
OUT = "/tmp/bench_gpu.mp4"
QUERY = "index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"
SAMPLE_START_S = 45.0
SAMPLE_END_S = 90.0

samples: list[tuple[float, list[str]]] = []
stop = threading.Event()


def sampler() -> None:
    t_start = time.time()
    while not stop.is_set():
        rel = time.time() - t_start
        if rel < SAMPLE_START_S:
            time.sleep(0.25)
            continue
        if rel > SAMPLE_END_S:
            stop.set()
            break
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            t = rel
            for line in out.stdout.strip().splitlines():
                samples.append((t, [p.strip() for p in line.split(",")]))
        except Exception as exc:  # noqa: BLE001
            print("sample failed:", exc, file=sys.stderr)
        time.sleep(0.5)


cmd = [
    "python",
    "-u",
    "inference.py",
    "--distilled-checkpoint-path",
    "checkpoints/avatarforever-ltx-2.3-22b.safetensors",
    "--gemma-root",
    "checkpoints/gemma-3-12b-it-qat-q4_0-unquantized",
    "--audio-path",
    "eng_test_0.wav",
    "--num-frames",
    NUM_FRAMES,
    "--output-path",
    OUT,
]

thread = threading.Thread(target=sampler, daemon=True)
thread.start()
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - t0
stop.set()
thread.join(timeout=3)

print("=" * 60)
print(f"inference rc={proc.returncode} elapsed={elapsed:.1f}s")
for line in proc.stdout.strip().splitlines()[-4:]:
    print("  ", line)
if proc.returncode != 0:
    print(proc.stderr[-2000:])
    sys.exit(1)

# group samples per GPU
per_gpu: dict[str, list[list[str]]] = {}
for _, row in samples:
    per_gpu.setdefault(row[0], []).append(row)

print("=" * 60)
print(f"total sample rows: {len(samples)} (window {SAMPLE_START_S}-{SAMPLE_END_S}s)")
timeline = sorted(((float(t), float(row[0]), float(row[1])) for t, row in samples), key=lambda x: x[0])
print("timeline (t_s, gpu, sm_util%):")
print("  " + " ".join(f"{t:.1f}:{g}:{u:.0f}" for t, g, u in timeline[::2]))

print("=" * 60)
for gpu, rows in sorted(per_gpu.items()):
    busy = [r for r in rows if float(r[3]) > 10_000]  # memory.used > 10 GB
    if not busy:
        print(f"GPU {gpu}: no busy samples")
        continue
    util = [float(r[1]) for r in busy]
    mem_util = [float(r[2]) for r in busy]
    power = [float(r[4]) for r in busy]
    clk = [float(r[5]) for r in busy]
    print(
        f"GPU {gpu}: busy samples={len(busy)}  sm_util avg={sum(util)/len(util):.1f}% "
        f"max={max(util):.0f}%  mem_util avg={sum(mem_util)/len(mem_util):.0f}%  "
        f"power avg={sum(power)/len(power):.0f}W max={max(power):.0f}W  sm_clock avg={sum(clk)/len(clk):.0f}MHz"
    )
    hist = {}
    for u in util:
        b = int(u // 10) * 10
        hist[b] = hist.get(b, 0) + 1
    print("   sm_util histogram:", {f"{k}-{k+10}%": hist[k] for k in sorted(hist)})
    print(f"   peak memory: {max(float(r[3]) for r in busy)/1024:.1f} GB")
