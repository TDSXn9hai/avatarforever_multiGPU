"""Measure how strongly the AR diffusion sampler amplifies a *tiny* numerical change.

Flips ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` (a pure
GEMM-accumulation detail, ~1 ulp of bf16) and then runs the *unmodified* inference
pipeline. Comparing the result against a normal run tells us how much of a TP change
is "expected amplification" rather than a bug.

Run: python ltx_tp/sensitivity_check.py <inference.py args...>
"""

from __future__ import annotations

import sys

import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

sys.path.insert(0, "/root/paddlejob/workspace/env_run")

import inference  # noqa: E402

if __name__ == "__main__":
    inference.main()
