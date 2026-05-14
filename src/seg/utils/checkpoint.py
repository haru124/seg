# src/ods/utils/checkpoint.py
import torch
import os
from pathlib import Path


def save_checkpoint(
    model, optimizer, scheduler, epoch: int, metrics: dict,
    exp_id: str, checkpoint_dir: str, top_k: int = 3
):
    """
    Saves with filename: {exp_id}_epoch{epoch:03d}_loss{loss:.4f}_mIoU{miou:.4f}.pth
    Keeps only top-k by mIoU.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    miou = metrics.get("mIoU", 0.0)
    loss = metrics.get("val_loss", 0.0)

    fname = f"{exp_id}_epoch{epoch:03d}_loss{loss:.4f}_mIoU{miou:.4f}.pth"
    fpath = ckpt_dir / fname

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "exp_id": exp_id,
    }
    torch.save(state, fpath)
    print(f"[Checkpoint] Saved → {fpath}")

    # Prune to top_k by mIoU
    all_ckpts = sorted(
        ckpt_dir.glob(f"{exp_id}_epoch*.pth"),
        key=lambda p: float(p.stem.split("mIoU")[-1]),
        reverse=True
    )
    for old in all_ckpts[top_k:]:
        old.unlink()
        print(f"[Checkpoint] Removed old: {old.name}")

    return str(fpath)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu") -> dict:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    print(f"[Checkpoint] Loaded ← {path}  (epoch {state['epoch']})")
    return state