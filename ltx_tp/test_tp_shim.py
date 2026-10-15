"""CPU smoke test for the TP shim: 2-rank (gloo) sharded forward vs single-process forward.

Uses a tiny LTXModel built from a shrunk copy of the real checkpoint config, so it
runs in seconds and needs no GPU. Any error in the sharding math shows up as a large
max-abs difference here instead of as a wrong-looking video later.

Run: python -m ltx_tp.test_tp_shim
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator

from ltx_tp.tp_shim import TPContext, shard_transformer_model_

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "avatarforever-ltx-2.3-22b.safetensors"


def tiny_config() -> dict:
    with safe_open(str(CHECKPOINT), framework="pt") as f:
        cfg = json.loads(f.metadata()["config"])
    cfg = copy.deepcopy(cfg)
    t = cfg["transformer"]
    t.update(
        {
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 32,  # video inner dim = 128
            "audio_num_attention_heads": 4,
            "audio_attention_head_dim": 16,  # audio inner dim = 64
            "in_channels": 16,
            "out_channels": 16,
            "audio_in_channels": 16,
            "audio_out_channels": 16,
            "video_channel_condition_in_channels": 16,
            "cross_attention_dim": 128,  # must equal the video inner dim (as in the real config: 4096)
            "audio_cross_attention_dim": 64,
            "caption_channels": 64,
        }
    )
    return cfg


def make_inputs(cfg: dict) -> tuple[Modality, Modality]:
    t = cfg["transformer"]
    video_ctx_dim = t["cross_attention_dim"]
    audio_ctx_dim = t["audio_cross_attention_dim"]
    in_ch = t["in_channels"]
    torch.manual_seed(1234)

    t_v, t_a, n_ctx = 6, 4, 5

    def grid_3d(tokens: int) -> torch.Tensor:
        """(1, 3, tokens, 2) patch grid bounds for video tokens (t, h, w)."""
        idx = torch.arange(tokens, dtype=torch.float32)
        t_b = torch.stack([idx // 4, idx // 4 + 1], dim=-1)
        h_b = torch.stack([(idx // 2) % 2, (idx // 2) % 2 + 1], dim=-1)
        w_b = torch.stack([idx % 2, idx % 2 + 1], dim=-1)
        return torch.stack([t_b, h_b, w_b], dim=0).unsqueeze(0)

    def grid_1d(tokens: int) -> torch.Tensor:
        idx = torch.arange(tokens, dtype=torch.float32)
        return torch.stack([idx, idx + 1], dim=-1).unsqueeze(0).unsqueeze(0)

    video = Modality(
        latent=torch.randn(1, t_v, in_ch),
        sigma=torch.tensor([0.7]),
        timesteps=torch.full((1, t_v), 0.7),
        positions=grid_3d(t_v),
        context=torch.randn(1, n_ctx, video_ctx_dim),
        enabled=True,
    )
    audio = Modality(
        latent=torch.randn(1, t_a, in_ch),
        sigma=torch.tensor([0.7]),
        timesteps=torch.full((1, t_a), 0.7),
        positions=grid_1d(t_a),
        context=torch.randn(1, n_ctx, audio_ctx_dim),
        enabled=True,
    )
    return video, audio


def build_model(cfg: dict):
    torch.manual_seed(0)
    model = LTXModelConfigurator.from_config(cfg)
    # The real weights come from the checkpoint; here we need sane values, because some
    # parameters are created with torch.empty (uninitialized memory ->可能含 NaN).
    with torch.no_grad():
        for param in model.parameters():
            if param.dtype.is_floating_point:
                param.normal_(0.0, 0.02)
        for buf in model.buffers():
            if buf.dtype.is_floating_point and not torch.isfinite(buf).all():
                buf.normal_(0.0, 0.02)
    return model.eval().to(torch.float32)


def _grab_tensor(out):
    """Pull a tensor out of whatever a module returned (tensor / tuple / TransformerArgs)."""
    if isinstance(out, torch.Tensor):
        return out.detach().clone()
    if isinstance(out, tuple):
        for item in out:
            found = _grab_tensor(item)
            if found is not None:
                return found
        return None
    x = getattr(out, "x", None)
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return None


def _capture_pass(model, video, audio, perturbations, targets: list[str]) -> dict:
    """Run one forward pass and capture outputs of the given module paths."""
    captured: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    handles = []
    for path in targets:
        module = modules.get(path)
        if module is None:
            continue
        def _hook(_m, _i, out, p=path):
            grabbed = _grab_tensor(out)
            if grabbed is not None:
                captured[p] = grabbed

        handles.append(module.register_forward_hook(_hook))
    with torch.no_grad():
        model(video, audio, perturbations)
    for handle in handles:
        handle.remove()
    return captured


def _patch_attention_capture(model, store: dict) -> None:
    """Wrap each Attention's attention_function to record (q, k, v, out) per pass."""
    from ltx_core.model.transformer.attention import Attention, PytorchAttention

    base = PytorchAttention()
    for path, module in model.named_modules():
        if not isinstance(module, Attention):
            continue

        def _make(p: str):
            def _call(q, k, v, heads, mask=None):
                store[(p, "q")] = q.detach().clone()
                store[(p, "k")] = k.detach().clone()
                store[(p, "v")] = v.detach().clone()
                out = base(q, k, v, heads, mask)
                store[(p, "out")] = out.detach().clone()
                return out

            return _call

        module.attention_function = _make(path)


def _compare_like(tp_tensor: torch.Tensor, ref_tensor: torch.Tensor, rank: int, world: int) -> float | None:
    """Max abs diff between a (possibly head-sharded) TP tensor and the reference."""
    if tp_tensor.shape == ref_tensor.shape:
        return (tp_tensor - ref_tensor).abs().max().item()
    if tp_tensor.shape[-1] * world == ref_tensor.shape[-1]:
        per = ref_tensor.shape[-1] // world
        ref_slice = ref_tensor[..., rank * per : (rank + 1) * per]
        return (tp_tensor - ref_slice).abs().max().item()
    return None


DEBUG_PATHS = [
    "transformer_blocks.0.attn1",
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn1.to_k",
    "transformer_blocks.0.attn1.q_norm",
    "transformer_blocks.0.attn1.to_out.0",
    "transformer_blocks.0.attn2",
    "transformer_blocks.0.audio_attn1",
    "transformer_blocks.0.audio_to_video_attn",
    "transformer_blocks.0.video_to_audio_attn",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.ff.net.2",
    "transformer_blocks.0.audio_ff.net.2",
    "transformer_blocks.1",
]


def run_one(rank: int, world: int, init_file: str, result_dir: str) -> None:
    # File-based rendezvous + file-based results: the sandbox forbids socket setup.
    init_url = Path(init_file).resolve().as_uri()
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init_url)
    tp = TPContext(rank=rank, world=world, device=torch.device("cpu"), verbose=(rank == 0))

    cfg = tiny_config()
    model = build_model(cfg)
    video, audio = make_inputs(cfg)
    perturbations = BatchedPerturbationConfig.empty(1)

    attn_store: dict = {}
    _patch_attention_capture(model, attn_store)

    ref_attn = _capture_pass(model, video, audio, perturbations, [])
    ref_attn = dict(attn_store)
    with torch.no_grad():
        ref_vx, ref_ax = model(video, audio, perturbations)
    ref_attn = dict(attn_store)
    ref_caps = _capture_pass(model, video, audio, perturbations, DEBUG_PATHS)

    shard_transformer_model_(model, tp)
    attn_store.clear()

    _ = _capture_pass(model, video, audio, perturbations, [])
    tp_attn = dict(attn_store)
    with torch.no_grad():
        tp_vx, tp_ax = model(video, audio, perturbations)
    tp_attn = dict(attn_store)
    tp_caps = _capture_pass(model, video, audio, perturbations, DEBUG_PATHS)

    attn_diffs = {}
    for (path, kind), ref_tensor in ref_attn.items():
        tp_tensor = tp_attn.get((path, kind))
        if tp_tensor is None:
            continue
        value = _compare_like(tp_tensor, ref_tensor, rank, world)
        attn_diffs[f"{path}.{kind}"] = ("n/a", float("nan")) if value is None else ("", value)

    diffs = {}
    for path, ref_tensor in ref_caps.items():
        tp_tensor = tp_caps.get(path)
        if tp_tensor is None:
            diffs[path] = ("missing", float("nan"))
            continue
        scale = max(ref_tensor.abs().max().item(), 1e-12)
        if tp_tensor.shape == ref_tensor.shape:
            diffs[path] = ("full", (ref_tensor - tp_tensor).abs().max().item() / scale)
            continue
        # sharded activation: compare this rank's slice of the reference
        if tp_tensor.shape[-1] * world == ref_tensor.shape[-1]:
            per = ref_tensor.shape[-1] // world
            ref_slice = ref_tensor[..., rank * per : (rank + 1) * per]
            diffs[path] = ("sliced", (ref_slice - tp_tensor).abs().max().item() / scale)
        else:
            diffs[path] = (
                f"shape {tuple(tp_tensor.shape)} vs {tuple(ref_tensor.shape)}",
                float("nan"),
            )

    dist.barrier()
    Path(result_dir, f"rank{rank}.json").write_text(
        json.dumps(
            {
                "vx_diff": (ref_vx - tp_vx).abs().max().item(),
                "ax_diff": (ref_ax - tp_ax).abs().max().item(),
                "vx_scale": ref_vx.abs().max().item(),
                "ax_scale": ref_ax.abs().max().item(),
                "diffs": diffs,
                "attn_diffs": attn_diffs,
            }
        )
    )
    dist.destroy_process_group()


def main() -> None:
    import sys

    world = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if world == 1:
        # Single-process sanity check: sharding with world=1 must be a bitwise no-op.
        tp = TPContext(rank=0, world=1, device=torch.device("cpu"), verbose=True)
        cfg = tiny_config()
        model = build_model(cfg)
        video, audio = make_inputs(cfg)
        perturbations = BatchedPerturbationConfig.empty(1)
        with torch.no_grad():
            ref_vx, ref_ax = model(video, audio, perturbations)
        shard_transformer_model_(model, tp)
        with torch.no_grad():
            tp_vx, tp_ax = model(video, audio, perturbations)
        print("=" * 60)
        print(f"world=1 video max|ref-tp1| = {(ref_vx - tp_vx).abs().max().item():.3e}")
        print(f"world=1 audio max|ref-tp1| = {(ref_ax - tp_ax).abs().max().item():.3e}")
        ok = (ref_vx - tp_vx).abs().max().item() == 0.0 and (ref_ax - tp_ax).abs().max().item() == 0.0
        print("RESULT:", "PASS (bitwise identical)" if ok else "FAIL")
        return

    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        mp.spawn(run_one, args=(world, init_file, tmp), nprocs=world, join=True)
        out = {r: json.loads(Path(tmp, f"rank{r}.json").read_text()) for r in range(world)}

    print("=" * 60)
    for rank, res in out.items():
        print(
            f"rank{rank}: video |ref|max={res['vx_scale']:.4f} max|ref-tp{world}|={res['vx_diff']:.3e} | "
            f"audio |ref|max={res['ax_scale']:.4f} max|ref-tp{world}|={res['ax_diff']:.3e}"
        )
        print(f"  rank{rank} per-module relative diff (vs matching reference slice):")
        for path, (kind, value) in sorted(res["diffs"].items(), key=lambda kv: -kv[1][1]):
            print(f"    {path:48s} [{kind:16s}] {value:.3e}")
        print(f"  rank{rank} attention internals abs diff (q/k/v/out):")
        for path, (_kind, value) in sorted(res["attn_diffs"].items()):
            print(f"    {path:60s} {value:.3e}")
    tol = 1e-4
    ok = all(r["vx_diff"] < tol and r["ax_diff"] < tol for r in out.values())
    print("RESULT:", "PASS" if ok else "FAIL", f"(tolerance {tol})")


if __name__ == "__main__":
    main()
