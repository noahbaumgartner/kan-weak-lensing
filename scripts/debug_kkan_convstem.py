"""Granular repro for the silent (no-traceback) death of the KKAN conv-stem
run: builds the real dataset + model via Hydra (same config as
scripts/test_kkan_convstem.submit) and runs a couple of real batches through
each stage by hand (stem -> conv1 -> pool1 -> conv2 -> pool1 -> kan1),
printing host RSS + CUDA memory after every stage with flush=True so we still
see progress if the process gets hard-killed (SIGKILL/OOM-killer) partway
through instead of raising a catchable Python exception.

Run directly (not via main.py) so nothing besides this script's own prints
can be lost: uv run python scripts/debug_kkan_convstem.py
"""
import resource
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.modules.reduction import reduced_dim


def _make_width(in_d, out_d, n_h, *hidden_widths):
    n_h = int(n_h)
    return [int(in_d)] + [int(w) for w in hidden_widths[:n_h]] + [int(out_d)]


OmegaConf.register_new_resolver("make_width", _make_width, replace=True)
OmegaConf.register_new_resolver("reduced_dim", reduced_dim, replace=True)


def mem(tag: str, device: torch.device) -> None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB on Linux
    line = f"[{tag}] host RSS(peak)={rss_gb:.2f}GB"
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        line += f" cuda_alloc={alloc:.2f}GB cuda_reserved={reserved:.2f}GB cuda_peak={peak:.2f}GB"
    print(line, flush=True)


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=kkan",
                "dataset=weak_lensing",
                "objective=mse",
                "training.batch_size=128",
            ],
        )

    from src import get_device
    device = get_device()
    print(f"device={device}", flush=True)
    mem("start", device)

    print("=== building dataset ===", flush=True)
    dataset = instantiate(cfg.dataset)
    data = dataset.create()
    mem("dataset.create() done", device)

    train_ds = data["train_input"]
    print(f"train_ds size={len(train_ds)}", flush=True)

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)

    print("=== building model ===", flush=True)
    model = instantiate(cfg.model)
    model.build(device=str(device))
    mem("model.build() done", device)
    print(f"parameter_count={model.parameter_count()}", flush=True)

    stem = model.model.stem
    inner = model.model.inner
    print(f"conv1 kernels={len(inner.conv1.convs)} conv2 kernels={len(inner.conv2.convs)}", flush=True)

    opt = torch.optim.Adam(model.model.parameters(), lr=1e-3)

    for step, (x, y) in enumerate(loader):
        if step >= 2:
            break
        print(f"--- step {step} ---", flush=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mem(f"step{step} batch to device (x={tuple(x.shape)})", device)

        x = model._prepare_input(x)
        mem(f"step{step} after _prepare_input (x={tuple(x.shape)})", device)

        h = stem(x)
        mem(f"step{step} after stem (h={tuple(h.shape)})", device)

        h = inner.conv1(h)
        mem(f"step{step} after conv1 (h={tuple(h.shape)})", device)
        h = inner.pool1(h)
        mem(f"step{step} after pool1 (h={tuple(h.shape)})", device)
        h = inner.conv2(h)
        mem(f"step{step} after conv2 (h={tuple(h.shape)})", device)
        h = inner.pool1(h)
        mem(f"step{step} after pool1#2 (h={tuple(h.shape)})", device)
        h = inner.flat(h)
        pred = inner.kan1(h)
        mem(f"step{step} after kan1 (pred={tuple(pred.shape)})", device)

        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        mem(f"step{step} after backward+step (loss={loss.item():.4f})", device)

    print("=== all steps completed without a hard kill ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("=== caught Python exception ===", flush=True)
        traceback.print_exc()
        raise
