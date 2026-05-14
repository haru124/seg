"""
src/seg/utils/checkpoint.py
-----------------------------
Save and load model checkpoints.

Checkpoint filename format:
    {exp_id}_epoch{epoch:03d}_loss{val_loss:.4f}_mIoU{miou:.4f}.pth

Only the top-k checkpoints (by val mIoU) are kept.
Older / lower-scoring checkpoints are automatically deleted.
"""

import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    exp_id: str,
    checkpoint_dir: str,
    top_k: int = 3,
) -> str:
    """
    Save a checkpoint and prune to keep only the top-k by val mIoU.

    Args:
        model          : nn.Module
        optimizer      : torch optimizer
        scheduler      : LR scheduler or None
        epoch          : current epoch number
        metrics        : dict from SegmentationMetrics.compute()
                         (must contain "mIoU" and "val_loss")
        exp_id         : experiment ID string (used in filename)
        checkpoint_dir : directory to save into
        top_k          : maximum number of checkpoints to keep

    Returns:
        Path string of the saved checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    miou     = metrics.get("mIoU",     0.0)
    val_loss = metrics.get("val_loss", 0.0)

    # Human-readable filename makes it easy to see quality at a glance
    fname = (
        f"{exp_id}_epoch{epoch:03d}"
        f"_loss{val_loss:.4f}"
        f"_mIoU{miou:.4f}.pth"
    )
    fpath = ckpt_dir / fname

    state = {
        "epoch"          : epoch,
        "exp_id"         : exp_id,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics"        : metrics,
    }

    torch.save(state, fpath)
    print(f"[Checkpoint] Saved → {fpath}")

    # ── Keep only top_k checkpoints ranked by mIoU ──
    all_ckpts = sorted(
        ckpt_dir.glob(f"{exp_id}_epoch*.pth"),
        key=_extract_miou,
        reverse=True,   # highest mIoU first
    )
    for old_ckpt in all_ckpts[top_k:]:
        old_ckpt.unlink()
        print(f"[Checkpoint] Pruned → {old_ckpt.name}")

    return str(fpath)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        path      : path to the .pth checkpoint file
        model     : nn.Module to load weights into
        optimizer : torch optimizer to restore state (or None)
        scheduler : LR scheduler to restore state (or None)
        device    : device string for map_location

    Returns:
        The full state dict (contains epoch, metrics, exp_id, etc.)
    """
    state = torch.load(path, map_location=device)

    model.load_state_dict(state["model_state"])

    if optimizer is not None and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])

    if scheduler is not None and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])

    print(
        f"[Checkpoint] Loaded ← {path}  "
        f"(epoch {state['epoch']}, "
        f"mIoU {state['metrics'].get('mIoU', 'N/A'):.4f})"
    )
    return state


# ── Helper ────────────────────────────────────────────────────────────

def _extract_miou(path: Path) -> float:
    """Parse mIoU value from checkpoint filename for sorting."""
    try:
        return float(path.stem.split("mIoU")[-1])
    except (IndexError, ValueError):
        return 0.0
