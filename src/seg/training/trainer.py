"""
src/seg/training/trainer.py
-----------------------------
Training loop for DeepLabV3+ on Cityscapes.

What gets logged where:
    Logger (file)  : every epoch — loss, mIoU, all scalar metrics
    TensorBoard    : scalars every epoch, per-class IoU as scalars,
                     confusion matrix image every cfg.viz.cm_interval epochs
    MLflow         : params once, scalars every epoch, per-class IoU,
                     confusion matrix PNG as artifact
    Disk           : confusion matrix PNG, per-class IoU bar chart
                     (outputs/viz/<exp_id>/)
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import numpy as np

from src.seg.evaluation.metrics import SegmentationMetrics, CITYSCAPES_CLASSES
from src.seg.utils.checkpoint import save_checkpoint, load_checkpoint
from src.seg.utils.common import setup_logger
from src.seg.tracking.mlflow_logger import MLflowLogger
from src.seg.tracking.tensorboard_logger import TensorboardLogger
from src.seg.entity.config_entity import ExperimentConfig

class EarlyStopping:
    """
    Stops training when monitored metric stops improving.

    Args:
        patience  : epochs to wait after last improvement
        min_delta : minimum change to count as improvement
        mode      : "max" for mIoU (higher=better), "min" for loss
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Call after each validation epoch.
        Returns True if training should stop.
        """
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop
    

class Trainer:
    """
    Manages the full train → validate → checkpoint → log cycle.

    Args:
        model        : DeepLabV3Plus model (not yet moved to device)
        optimizer    : torch optimizer
        scheduler    : LR scheduler or None
        loss_fn      : loss module
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

        self.scaler = GradScaler(enabled=cfg.training.amp)

        self.metrics = SegmentationMetrics(
            num_classes=cfg.data.num_classes,
            ignore_index=cfg.data.ignore_index,
            compute_boundary= False
        )

        self.logger = setup_logger(
            "trainer",
            log_dir=str(Path(cfg.checkpoint.dir).parent / "logs"),
            exp_id=cfg.experiment_id,
        )

        self.tb = (
            TensorboardLogger(cfg.tracking.tb_log_dir, cfg.experiment_id)
            if cfg.tracking.tb_enabled else None
        )

        self.mlf = (
            MLflowLogger(
                cfg.tracking.mlflow_uri,
                cfg.tracking.mlflow_experiment,
                run_name=cfg.experiment_id,
            )
            if cfg.tracking.mlflow_enabled else None
        )

        # Viz output directory
        self.viz_dir = Path(cfg.checkpoint.dir).parent / "viz" / cfg.experiment_id
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # How often to save confusion matrix (every N epochs)
        self.cm_interval = 5

        # Resume
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
        # ── Early stopping ──
        patience    = getattr(cfg.training, "early_stopping_patience",  10)
        min_delta   = getattr(cfg.training, "early_stopping_min_delta", 0.001)
        self.early_stopping = EarlyStopping(
                patience=patience,
                min_delta=min_delta,
                mode="max",   # monitoring mIoU — higher is better
            )
        
        self.logger.info(
            f"[EarlyStopping] patience={patience}, min_delta={min_delta}, monitor=mIoU"
        )

    # ── Train one epoch ────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        accum      = self.cfg.training.accumulation_steps

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

            with autocast(device_type=self.device.type,
                          enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss    = self.loss_fn(outputs, targets) / accum

            self.scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(self.train_loader):
                if self.cfg.training.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            batch_loss  = loss.item() * accum
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        return total_loss / len(self.train_loader)

    # ── Validate one epoch ─────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        self.metrics.compute_boundary = (epoch%5 == 0)

        pbar = tqdm(
            self.val_loader,
            desc=f"[Epoch {epoch:03d}]   Val",
            leave=False,
        )

        for images, targets in pbar:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            with autocast(device_type=self.device.type,
                          enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss    = self.loss_fn(outputs, targets)

            pred_logits = outputs["out"] if isinstance(outputs, dict) else outputs
            preds       = pred_logits.argmax(dim=1)

            self.metrics.update(preds, targets)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        result             = self.metrics.compute()
        result["val_loss"] = total_loss / len(self.val_loader)
        return result

    # ── Logging helpers ────────────────────────────────────────────────

    def _log_scalars(self, val_metrics: dict, train_loss: float, epoch: int):
        """Log all scalar metrics to logger file, TensorBoard, MLflow."""
        num_classes  = self.cfg.data.num_classes
        class_names  = CITYSCAPES_CLASSES[:num_classes]

        # ── File logger ────────────────────────────────────────────────
        self.logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"mIoU={val_metrics['mIoU']:.4f} | "
            f"fw_iou={val_metrics['fw_iou']:.4f} | "
            f"px_acc={val_metrics['mean_pixel_acc']:.4f} | "
            f"prec={val_metrics['mean_precision']:.4f} | "
            f"recall={val_metrics['mean_recall']:.4f} | "
            f"f1={val_metrics['mean_f1']:.4f} | "
            f"b_iou={val_metrics['boundary_iou']:.4f}"
        )
        if epoch %5 == 0:

            # Per-class IoU to file logger
            per_iou = val_metrics.get("per_class_iou", [])
            if per_iou:
                iou_strs = [
                    f"  {name:<20}: {v:.4f}" if not np.isnan(v) else f"  {name:<20}: N/A"
                    for name, v in zip(class_names, per_iou)
                ]
                self.logger.info("Per-class IoU:\n" + "\n".join(iou_strs))

        # ── TensorBoard ────────────────────────────────────────────────
        if self.tb:
            # Loss
            self.tb.log_scalar("Loss/train",           train_loss,                          epoch)
            if val_metrics:
                self.tb.log_scalar("Loss/val",             val_metrics["val_loss"],             epoch)

                # Overall metrics
                self.tb.log_scalar("Metrics/mIoU",         val_metrics["mIoU"],                 epoch)
                self.tb.log_scalar("Metrics/fw_iou",       val_metrics["fw_iou"],               epoch)
                self.tb.log_scalar("Metrics/pixel_acc",    val_metrics["mean_pixel_acc"],        epoch)
                self.tb.log_scalar("Metrics/class_acc",    val_metrics["mean_class_acc"],        epoch)
                self.tb.log_scalar("Metrics/precision",    val_metrics["mean_precision"],        epoch)
                self.tb.log_scalar("Metrics/recall",       val_metrics["mean_recall"],           epoch)
                self.tb.log_scalar("Metrics/f1",           val_metrics["mean_f1"],               epoch)
                
                #self.tb.log_scalar("Metrics/boundary_iou", val_metrics["boundary_iou"],          epoch)
                #self.tb.log_scalar("Metrics/boundary_f",   val_metrics["boundary_fscore"],       epoch)
                # ✅ Per-class metrics: LOG ONLY EVERY 5 EPOCHS
                if epoch % 5 == 0:
                    per_class_iou = val_metrics["per_class_iou"]
                    for cls_idx, (cls_name, iou) in enumerate(
                        zip(CITYSCAPES_CLASSES, per_class_iou)
                    ):
                        if not np.isnan(iou):
                            self.tb.log_scalar(f"PerClass_IoU/{cls_name}", iou, epoch)
            
            # LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.tb.log_scalar("LR", current_lr, epoch)

        # ── MLflow ─────────────────────────────────────────────────────
        # ── MLflow logging ──
        if self.mlf:
            log_dict = {
                "train_loss": train_loss,
            }
            
            if val_metrics:
                # Main metrics
                log_dict.update({
                    "val_loss": val_metrics["val_loss"],
                    "mIoU": val_metrics["mIoU"],
                    "fw_iou": val_metrics["fw_iou"],
                    "pixel_acc": val_metrics["mean_pixel_acc"],
                    "class_acc": val_metrics["mean_class_acc"],
                    "precision": val_metrics["mean_precision"],
                    "recall": val_metrics["mean_recall"],
                    "f1": val_metrics["mean_f1"],
                    "boundary_iou": val_metrics["boundary_iou"],
                    "boundary_fscore": val_metrics["boundary_fscore"],
                })
                
                # Per-class metrics: LOG ONLY EVERY 5 EPOCHS
                if epoch % 5 == 0:
                    for cls_idx, cls_name in enumerate(CITYSCAPES_CLASSES):
                        iou = val_metrics["per_class_iou"][cls_idx]
                        precision = val_metrics["per_class_precision"][cls_idx]
                        recall = val_metrics["per_class_recall"][cls_idx]
                        
                        if not np.isnan(iou):
                            log_dict[f"iou_{cls_name}"] = float(iou)
                        if not np.isnan(precision):
                            log_dict[f"precision_{cls_name}"] = float(precision)
                        if not np.isnan(recall):
                            log_dict[f"recall_{cls_name}"] = float(recall)
            
            self.mlf.log_metrics(log_dict, step=epoch)
        
        
    def _save_visualizations(self, val_metrics: dict, epoch: int):
        """Save confusion matrix PNG and per-class IoU bar chart."""
        from src.seg.utils.visualization import save_confusion_matrix, plot_class_iou

        num_classes = self.cfg.data.num_classes
        class_names = CITYSCAPES_CLASSES[:num_classes]

        # ── Confusion matrix ───────────────────────────────────────────
        #  Save confusion matrix: only at key epochs
        if epoch % 10 == 0 or epoch == self.cfg.training.epochs:
            from src.seg.utils.visualization import save_confusion_matrix
            cm_path = (Path(self.cfg.checkpoint.dir).parent / "confusion_matrices" / 
                        f"{self.cfg.experiment_id}_epoch{epoch:03d}_cm.png")
            save_confusion_matrix(
                val_metrics["confusion_matrix"],
                str(cm_path),
                normalize=True,
            )
        # Log to MLflow as artifact
        if self.mlf:
            try:
                self.mlf.log_artifact(str(cm_path))
            except Exception:
                pass  # mlflow artifact logging is optional

        # ── Per-class IoU bar chart ────────────────────────────────────
        iou_path = self.viz_dir / f"per_class_iou_epoch{epoch:03d}.png"
        plot_class_iou(
            val_metrics["per_class_iou"],
            str(iou_path),
            title=f"Per-Class IoU — Epoch {epoch:03d} | mIoU={val_metrics['mIoU']:.4f}",
            class_names=class_names,
        )
        if self.mlf:
            try:
                self.mlf.log_artifact(str(iou_path))
            except Exception:
                pass

    # ── Main training loop ─────────────────────────────────────────────

    def train(self):
        cfg = self.cfg

        if self.mlf:
            self.mlf.log_params({
                "backbone"       : cfg.model.backbone,
                "output_stride"  : cfg.model.output_stride,
                "loss"           : cfg.loss.type,
                "optimizer"      : cfg.training.optimizer,
                "lr"             : cfg.training.lr,
                "epochs"         : cfg.training.epochs,
                "batch_size"     : cfg.data.batch_size,
                "accum_steps"    : cfg.training.accumulation_steps,
                "amp"            : cfg.training.amp,
                "aux_loss"       : cfg.training.aux_loss,
                "image_size"     : str(cfg.data.image_size),
            })

        self.logger.info(
            f"Starting training: exp={cfg.experiment_id}  "
            f"epochs={cfg.training.epochs}  device={self.device}"
        )

        eval_interval = getattr(cfg, "evaluation", None)
        eval_interval = eval_interval.interval if eval_interval else 1

        for epoch in range(self.start_epoch, cfg.training.epochs + 1):

            # ── Train ──────────────────────────────────────────────────
            train_loss = self._train_epoch(epoch)

            # ── Validate ───────────────────────────────────────────────
            val_metrics = {}
            if epoch % eval_interval == 0:
                val_metrics                = self._val_epoch(epoch)
                val_metrics["train_loss"] = train_loss

                # Log all scalars + per-class
                self._log_scalars(val_metrics, train_loss, epoch)

                # Save confusion matrix + IoU chart every cm_interval epochs
                if epoch % self.cm_interval == 0:
                    self._save_visualizations(val_metrics, epoch)

            else:
                self.logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")
                if self.tb:
                    self.tb.log_scalar("Loss/train", train_loss, epoch)
                if self.mlf:
                    self.mlf.log_metrics({"train_loss": train_loss}, step=epoch)

            # ── LR step ────────────────────────────────────────────────
            if self.scheduler is not None:
                self.scheduler.step()

            # ── Checkpoint ─────────────────────────────────────────────
            if val_metrics:
                save_checkpoint(
                    model          = self.model,
                    optimizer      = self.optimizer,
                    scheduler      = self.scheduler,
                    epoch          = epoch,
                    metrics        = val_metrics,
                    exp_id         = cfg.experiment_id,
                    checkpoint_dir = cfg.checkpoint.dir,
                    top_k          = cfg.checkpoint.save_top_k,
                )
        # ── Early stopping ─────────────────────────────────────────
            if val_metrics:
                current_miou = val_metrics["mIoU"]
                stop = self.early_stopping.step(current_miou)

                self.logger.info(
                    f"[EarlyStopping] mIoU={current_miou:.4f} | "
                    f"best={self.early_stopping.best_value:.4f} | "
                    f"counter={self.early_stopping.counter}/{self.early_stopping.patience}"
                )

                if stop:
                    self.logger.info(
                        f"[EarlyStopping] Triggered at epoch {epoch} — "
                        f"no improvement for {self.early_stopping.patience} epochs. "
                        f"Best mIoU: {self.early_stopping.best_value:.4f}"
                    )
                    # Save final visualizations before stopping
                    self._save_visualizations(val_metrics, epoch)
                    break   # exits the epoch loop cleanly

            # ── Cleanup ────────────────────────────────────────────────────
            # Save final confusion matrix regardless of cm_interval
        if val_metrics:
            self._save_visualizations(val_metrics, epoch)

        if self.tb:
            self.tb.close()
        if self.mlf:
            self.mlf.finish()

        self.logger.info("Training complete.")