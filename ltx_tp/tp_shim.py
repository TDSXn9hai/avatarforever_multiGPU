"""Minimal tensor-parallel (TP) shim for the Avatar-Forever / LTX diffusion transformer.

Layout: the hidden/residual stream stays **replicated** on every rank and only the
per-token ``nn.Linear`` weights are sharded (Megatron-style TP):

* column-parallel (output dim split by attention head, or by FFN inner dim) -> no communication;
* row-parallel (input dim split) -> outputs are summed with one ``all_reduce``.

Because the hidden stream is replicated, everything that operates on it
(non-affine ``rms_norm``/``LayerNorm``, AdaLN scale/shift tables, ``patchify_proj``,
``proj_out``, channel conditioning) needs no change at all. Only two things do:

1. ``q_norm`` / ``k_norm`` are RMSNorm over the *full* attention inner dim, but each
   rank only holds its own heads -> the sum of squares must be reduced globally.
2. attention head count and the RoPE tables (``pe``/``k_pe``) must be sliced to the
   local head range.

Not supported: perturbation masks that are neither fully on nor fully off (the ones
used by the pipeline are always ``None`` in the released inference path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TPContext:
    """Rank/world info plus the collective helpers used by the shim."""

    rank: int
    world: int
    device: torch.device
    group: Any = None
    verbose: bool = False

    def block(self, size: int) -> slice:
        """Contiguous block of ``size`` belonging to this rank."""
        if size % self.world != 0:
            raise ValueError(f"dimension {size} is not divisible by tp world size {self.world}")
        per = size // self.world
        return slice(self.rank * per, (self.rank + 1) * per)

    def all_reduce_sum_(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor


class TPLinear(nn.Module):
    """A Linear holding only this rank's slice of the original weight.

    ``reduce_output`` must be True for row-parallel layers (input dim sharded), whose
    partial products have to be summed across ranks.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, reduce_output: bool, tp: TPContext):
        super().__init__()
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.bias: torch.Tensor | None = None
        if bias is not None:
            self.register_buffer("weight_bias", bias.detach().contiguous(), persistent=False)
            self.bias = self.weight_bias
        self.reduce_output = reduce_output
        self.tp = tp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.reduce_output or self.tp.world == 1:
            # Plain / column-parallel layer: identical to the original fused linear.
            return F.linear(x, self.weight, self.bias)

        # Row-parallel: sum partials *before* the bias (adding it per rank would count it twice).
        # The reduction and the bias add run in fp32 to keep the rounding close to cuBLAS' fused
        # epilogue of the unsharded layer.
        partial = F.linear(x, self.weight, None)
        acc = partial.float() if partial.dtype in (torch.bfloat16, torch.float16) else partial
        self.tp.all_reduce_sum_(acc)
        if self.bias is not None:
            acc = acc + self.bias.float()
        return acc.to(partial.dtype)

    def extra_repr(self) -> str:
        return f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, reduce={self.reduce_output}"


class TPRMSNorm(nn.Module):
    """RMSNorm over a dimension that is split across ranks (q_norm / k_norm)."""

    def __init__(self, weight: torch.Tensor, eps: float, tp: TPContext):
        super().__init__()
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.eps = eps
        self.tp = tp
        self.full_dim = int(weight.numel()) * tp.world

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror torch.nn.functional.rms_norm: accumulate in fp32 and round only once at the end.
        xf = x.float()
        sumsq = xf.pow(2).sum(dim=-1, keepdim=True)
        self.tp.all_reduce_sum_(sumsq)
        inv_rms = torch.rsqrt(sumsq / self.full_dim + self.eps)
        out = xf * inv_rms
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(x.dtype)


# --- module-level patches -------------------------------------------------------------

_ORIG_ATTENTION_FORWARD = None


def _slice_pe(pe: Any, head_slice: slice, full_heads: int) -> Any:
    """Slice precomputed RoPE tables along their head axis (B, H, T, D//2)."""

    def _one(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4 and t.shape[1] == full_heads:
            return t[:, head_slice]
        if t.ndim == 4 and t.shape[-1] == full_heads:  # (B, T, H, D//2) layout, defensive
            return t[..., head_slice, :]
        return t

    if isinstance(pe, (tuple, list)):
        return tuple(_one(t) for t in pe)
    return _one(pe)


def _tp_attention_forward(
    self,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pe: Any = None,
    k_pe: Any = None,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool = False,
) -> torch.Tensor:
    if perturbation_mask is not None and not all_perturbed:
        raise NotImplementedError(
            "Tensor parallelism requires perturbations to be either fully on or fully off "
            "(the released inference path never sets a partial perturbation mask)."
        )
    if pe is not None:
        pe = _slice_pe(pe, self.tp_head_slice, self.tp_full_heads)
    if k_pe is not None:
        k_pe = _slice_pe(k_pe, self.tp_head_slice, self.tp_full_heads)
    return _ORIG_ATTENTION_FORWARD(
        self,
        x,
        context=context,
        mask=mask,
        pe=pe,
        k_pe=k_pe,
        perturbation_mask=perturbation_mask,
        all_perturbed=all_perturbed,
    )


# --- weight sharding ------------------------------------------------------------------

_COL_SUFFIXES = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_gate_logits",
    ".ff.net.0.proj",
    ".audio_ff.net.0.proj",
)
_ROW_SUFFIXES = (
    ".to_out.0",
    ".ff.net.2",
    ".audio_ff.net.2",
)


def _replacement_for(path: str, lin: nn.Linear, tp: TPContext) -> tuple[str, slice | None, slice | None] | None:
    """Return (mode, out_slice, in_slice) for a Linear, or None to keep it replicated."""
    if path.endswith(_COL_SUFFIXES):
        return "col", tp.block(lin.weight.shape[0]), None
    if path.endswith(_ROW_SUFFIXES):
        return "row", None, tp.block(lin.weight.shape[1])
    return None


def _shard_linear(lin: nn.Linear, mode: str, out_slice: slice | None, in_slice: slice | None, tp: TPContext) -> TPLinear:
    weight = lin.weight.data
    bias = lin.bias.data if lin.bias is not None else None
    if mode == "col":
        weight = weight[out_slice]
        if bias is not None:
            bias = bias[out_slice]
        reduce_output = False
    else:  # row
        weight = weight[:, in_slice]
        reduce_output = True
    return TPLinear(weight, bias, reduce_output, tp)


def shard_transformer_model_(model: nn.Module, tp: TPContext) -> nn.Module:
    """Shard every per-token Linear of the (already loaded, single-GPU) DiT in place."""
    global _ORIG_ATTENTION_FORWARD

    from ltx_core.model.transformer.attention import Attention

    if _ORIG_ATTENTION_FORWARD is None:
        _ORIG_ATTENTION_FORWARD = Attention.forward

    sharded = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            path = f"{name}.{child_name}" if name else child_name
            plan = _replacement_for(path, child, tp)
            if plan is None:
                continue
            mode, out_slice, in_slice = plan
            setattr(module, child_name, _shard_linear(child, mode, out_slice, in_slice, tp))
            sharded += 1

    # per-attention hooks: local head count, sliced RoPE tables, globally normalized q/k
    attn_count = 0
    for _, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        attn_count += 1
        full_heads = module.heads
        head_slice = tp.block(full_heads)
        inner_slice = slice(head_slice.start * module.dim_head, head_slice.stop * module.dim_head)

        module.tp_full_heads = full_heads
        module.tp_head_slice = head_slice
        module.heads = full_heads // tp.world

        for norm_name in ("q_norm", "k_norm"):
            norm = getattr(module, norm_name)
            weight = norm.weight.data[inner_slice]
            setattr(module, norm_name, TPRMSNorm(weight, norm.eps, tp))

        module.forward = _tp_attention_forward.__get__(module, type(module))

    if tp.verbose and tp.rank == 0:
        print(f"[tp] sharded {sharded} Linear layers, patched {attn_count} attention modules", flush=True)
    return model
