"""
src/seg/training/trainer.py
-----------------------------
Training loop for DeepLabV3+ on Cityscapes.

Features:
  - Mixed precision (AMP) via torch.cuda.amp
  - Gradient accumulation (effective large batch on 4 GB GPU)
  - Optional gradient clipping
  - Checkpointing: saves top-k by val mIoU
  - MLflow + TensorBoard logging
  - Resume from checkpoint
  - Poly / cosine / step LR schedulers

The Trainer only knows about the standard interface:
    model output  → dict {"out": tensor} or {"out": tensor, "aux": tensor}
    loss_fn input → (model_output_dict, target)
    metrics.update(pred_argmax, target)
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from src.seg.evaluation.metrics import SegmentationMetrics
from src.seg.utils.checkpoint import save_checkpoint, load_checkpoint
from src.seg.utils.common import setup_logger
from src.seg.tracking.mlflow_logger import MLflowLogger
from src.seg.tracking.tensorboard_logger import TensorboardLogger
from src.seg.entity.config_entity import ExperimentConfig
from src.seg.constants import CITYSCAPES_CLASSES


class Trainer:
    """
    Manages the full train → validate → checkpoint → log cycle.

    Args:
        model        : DeepLabV3Plus model (not yet moved to device)
        optimizer    : torch optimizer
        scheduler    : LR scheduler or None
        loss_fn      : loss module — accepts (dict_output, target)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        cfg          : ExperimentConfig
        device       : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        loss_fn: nn.Module,
        train_loader,
        val_loader,
        cfg: ExperimentConfig,
        device: torch.device,
    ):
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        # AMP scaler — does nothing if amp=False
        self.scaler = GradScaler(enabled=cfg.training.amp)

        # Metrics tracker
        self.metrics = SegmentationMetrics(
            num_classes=cfg.data.num_classes,
            ignore_index=cfg.data.ignore_index,
        )

        # Logger (file + console)
        self.logger = setup_logger(
            "trainer",
            log_dir=str(Path(cfg.checkpoint.dir).parent / "logs"),
            exp_id=cfg.experiment_id,
        )

        # TensorBoard
        self.tb = (
            TensorboardLogger(cfg.tracking.tb_log_dir, cfg.experiment_id)
            if cfg.tracking.tb_enabled else None
        )

        # MLflow
        self.mlf = (
            MLflowLogger(
                cfg.tracking.mlflow_uri,
                cfg.tracking.mlflow_experiment,
                run_name=cfg.experiment_id,
            )
            if cfg.tracking.mlflow_enabled else None
        )

        # ── Resume from checkpoint if requested ──
        self.start_epoch = 1
        if cfg.checkpoint.resume:
            state = load_checkpoint(
                cfg.checkpoint.resume, model, optimizer, scheduler, str(device)
            )
            self.start_epoch = state["epoch"] + 1
            self.logger.info(
                f"Resumed from epoch {state['epoch']} — "
                f"starting at epoch {self.start_epoch}"
            )

    # ── Train one epoch ───────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """Run one full training epoch. Returns average train loss."""
        self.model.train()
        total_loss = 0.0
        accum = self.cfg.training.accumulation_steps

        self.optimizer.zero_grad()
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"[Epoch {epoch:03d}] Train",
            leave=False,
        )

        for i, (images, targets) in pbar:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            # Forward pass under AMP context
            with autocast(enabled=self.cfg.training.amp):
                outputs = self.model(images)
                # Divide by accum so gradients average correctly
                loss = self.loss_fn(outputs, targets) / accum

            # Backward under AMP scaler
            self.scaler.scale(loss).backward()

            # Gradient step every `accum` mini-batches
            if (i + 1) % accum == 0 or (i + 1) == len(self.train_loader):
                if self.cfg.training.grad_clip:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # loss * accum to recover the actual unscaled loss value
            batch_loss = loss.item() * accum
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        return total_loss / len(self.train_loader)

    # ── Validate one epoch ────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        """Run full validation. Returns metrics dict."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        pbar = tqdm(
            self.val_loader,
            desc=f"[Epoch {epoch:03d}]   Val",
            leave=False,
        )

        for images, targets in pbar:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            with autocast(enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss    = self.loss_fn(outputs, targets)

            # Extract main prediction for metrics
            pred_logits = outputs["out"] if isinstance(outputs, dict) else outputs
            preds = pred_logits.argmax(dim=1)   # (B, H, W) long

            self.metrics.update(preds, targets)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        result = self.metrics.compute()
        result["val_loss"] = total_loss / len(self.val_loader)
        return result

    # ── Main training loop ────────────────────────────────────────────

    def train(self):
        """Run training for all epochs, with logging and checkpointing."""
        cfg = self.cfg

        # Log hyperparameters to MLflow at the start
        if self.mlf:
            self.mlf.log_params({
                "backbone"        : cfg.model.backbone,
                "output_stride"   : cfg.model.output_stride,
                "loss"            : cfg.loss.type,
                "optimizer"       : cfg.training.optimizer,
                "lr"              : cfg.training.lr,
                "epochs"          : cfg.training.epochs,
                "batch_size"      : cfg.data.batch_size,
                "accum_steps"     : cfg.training.accumulation_steps,
                "amp"             : cfg.training.amp,
                "aux_loss"        : cfg.training.aux_loss,
                "image_size"      : cfg.data.image_size,
            })

        self.logger.info(
            f"Starting training: exp={cfg.experiment_id}  "
            f"epochs={cfg.training.epochs}  device={self.device}"
        )

        for epoch in range(self.start_epoch, cfg.training.epochs + 1):

            # ── Training ──
            train_loss = self._train_epoch(epoch)

            # ── Validation (every eval_interval epochs) ──
            val_metrics = {}
            eval_interval = getattr(cfg, "evaluation", None)
            eval_interval = eval_interval.interval if eval_interval else 1
            if epoch % eval_interval == 0:
                val_metrics = self._val_epoch(epoch)
                val_metrics["train_loss"] = train_loss

                self.logger.info(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['val_loss']:.4f} | "
                    f"mIoU={val_metrics['mIoU']:.4f} | "
                    f"fw_iou={val_metrics['fw_iou']:.4f} | "
                    f"px_acc={val_metrics['mean_pixel_acc']:.4f} | "
                    f"b_iou={val_metrics['boundary_iou']:.4f}"
                )
            else:
                # Only log train loss when not validating
                self.logger.info(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f}"
                )

            # ── LR scheduler step ──
            if self.scheduler is not None:
                self.scheduler.step()

            # ── TensorBoard logging ──
            if self.tb:
                self.tb.log_scalar("Loss/train", train_loss, epoch)
                if val_metrics:
                    self.tb.log_scalar("Loss/val",            val_metrics["val_loss"],        epoch)
                    self.tb.log_scalar("Metrics/mIoU",        val_metrics["mIoU"],            epoch)
                    self.tb.log_scalar("Metrics/fw_iou",      val_metrics["fw_iou"],          epoch)
                    self.tb.log_scalar("Metrics/pixel_acc",   val_metrics["mean_pixel_acc"],  epoch)
                    self.tb.log_scalar("Metrics/class_acc",   val_metrics["mean_class_acc"],  epoch)
                    self.tb.log_scalar("Metrics/b_iou",       val_metrics["boundary_iou"],    epoch)
                    self.tb.log_scalar("Metrics/b_fscore",    val_metrics["boundary_fscore"], epoch)
                # Log current LR
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.tb.log_scalar("LR", current_lr, epoch)

            # ── MLflow logging ──
            if self.mlf:
                log_dict = {"train_loss": train_loss}
                if val_metrics:
                    # Only log float values (skip per_class lists)
                    for k, v in val_metrics.items():
                        if isinstance(v, float):
                            log_dict[k] = v
                self.mlf.log_metrics(log_dict, step=epoch)

            # ── Save checkpoint ──
            if val_metrics:
                save_checkpoint(
                    model         = self.model,
                    optimizer     = self.optimizer,
                    scheduler     = self.scheduler,
                    epoch         = epoch,
                    metrics       = val_metrics,
                    exp_id        = cfg.experiment_id,
                    checkpoint_dir= cfg.checkpoint.dir,
                    top_k         = cfg.checkpoint.save_top_k,
                )

        # ── Cleanup ──
        if self.tb:
            self.tb.close()
        if self.mlf:
            self.mlf.finish()

        self.logger.info("Training complete.")