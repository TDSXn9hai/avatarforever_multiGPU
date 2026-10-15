"""Minimal 2-rank NCCL sanity check (run with: torchrun --nproc_per_node=2 ltx_tp/nccl_check.py)."""

import os

import torch
import torch.distributed as dist

_probe = os.environ.get("LTX_TP_PROBE_IMPORT")
if _probe:
    import importlib

    for _name in _probe.split(","):
        importlib.import_module(_name.strip())
    print(f"[probe] imported: {_probe}", flush=True)


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    print(f"[rank {rank}] cuda device set: {torch.cuda.get_device_name(rank)}", flush=True)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    print(f"[rank {rank}] process group ready", flush=True)
    if os.environ.get("LTX_TP_PROBE_AFTER"):
        import importlib

        for name in os.environ["LTX_TP_PROBE_AFTER"].split(","):
            importlib.import_module(name.strip())
        print(f"[rank {rank}] late import ok", flush=True)

    tensor = torch.ones(4, device=torch.device("cuda", rank))
    dist.all_reduce(tensor)
    print(f"[rank {rank}] all_reduce -> {tensor[0].item()}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()
