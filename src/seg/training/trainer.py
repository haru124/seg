"""
src/seg/training/trainer.py
-----------------------------
Training loop for DeepLabV3+ on Cityscapes.

What gets logged where:
    Logger (file)  : every epoch — loss, mIoU, all scalar metrics
    TensorBoard    : scalars every epoch, per-class IoU as scalars,
                        
    MLflow         : params once, scalars every epoch, per-class IoU,
                     
"""

import torch
import time
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
from datetime import datetime


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
        self.run_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        self.scaler = GradScaler(enabled=cfg.training.amp)

        self.metrics = SegmentationMetrics(
            num_classes=cfg.data.num_classes,
            ignore_index=cfg.data.ignore_index,
            compute_boundary= False
        )

        self.logger = setup_logger(
            "trainer",
            log_dir=str(Path(cfg.checkpoint.dir).parent / "logs" / cfg.experiment_id)  #"outputs/checkpoints".parent = outputs 
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

        # ── Training history ──────────────────────────────────────
        self.history = {
            "experiment_id": cfg.experiment_id,
            "run_timestamp": self.run_timestamp,
            "train": {
                "epoch": [],
                "loss": [],
            },
            "val": {
                "epoch": [],
                "loss": [],
                "miou": [],
                "pixel_acc": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "per_class_iou": [],
            }
        }

    
        # exp1_resnet_sgd_focal → exp1
        exp_stem = cfg.experiment_id.split("_")[0]

        # outputs/history/exp1/20260524_142355/
        self.history_dir = (
            Path(cfg.checkpoint.dir).parent
            / "history"
            / exp_stem
        )

        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.history_dir / f"{cfg.experiment_id}_{self.run_timestamp}_training_history.json"


    # ── Train one epoch ────────────────────────────────────────────────
    
    def _train_epoch(self, epoch: int):
        start_time = time.time()
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
                if self.scheduler is not None:   #LR step
                    self.scheduler.step()
                self.optimizer.zero_grad()

            batch_loss  = loss.item() * accum
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
        
        epoch_time = time.time() - start_time
        return total_loss / len(self.train_loader), epoch_time

    # ── Validate one epoch ─────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        val_start_time = time.time()
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
        val_time = time.time() - val_start_time
        result["val_time"] = val_time
        
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
            f"train_time={val_metrics['train_time']:.1f}s | "
            f"val_time={val_metrics['val_time']:.1f}s"
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
        

    def _save_history(self):
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

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
            train_loss, train_time = self._train_epoch(epoch)

            # ── Validate ───────────────────────────────────────────────
            val_metrics = {}
            if epoch % eval_interval == 0:
                val_metrics                = self._val_epoch(epoch)
                val_metrics["train_loss"] = train_loss

                val_metrics["train_time"] = train_time
                
                # ── Save history ───────────────────────────────────
                self.history["train"]["epoch"].append(epoch)
                self.history["train"]["loss"].append(train_loss)

                self.history["val"]["epoch"].append(epoch)
                self.history["val"]["loss"].append(
                    val_metrics["val_loss"]
                )

                self.history["val"]["miou"].append(
                    val_metrics["mIoU"]
                )

                self.history["val"]["pixel_acc"].append(
                    val_metrics["mean_pixel_acc"]
                )

                self.history["val"]["precision"].append(
                    val_metrics["mean_precision"]
                )

                self.history["val"]["recall"].append(
                    val_metrics["mean_recall"]
                )

                self.history["val"]["f1"].append(
                    val_metrics["mean_f1"]
                )

                self.history["val"]["per_class_iou"].append(
                    val_metrics["per_class_iou"]   
                )

                self._save_history()

                # Log all scalars + per-class
                self._log_scalars(val_metrics, train_loss, epoch)


            else:
                self.logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")
                if self.tb:
                    self.tb.log_scalar("Loss/train", train_loss, epoch)
                if self.mlf:
                    self.mlf.log_metrics({"train_loss": train_loss}, step=epoch)

            # ── Checkpoint ─────────────────────────────────────────────
            if val_metrics:
                """
                metrics_to_save = {
                    k: v for k, v in val_metrics.items()
                    if k not in ("confusion_matrix", "per_class_iou", "per_class_precision",
                                "per_class_recall", "per_class_f1", "per_class_support",
                                "per_class_boundary_iou", "per_class_boundary_fscore")
                }
                """
                metrics_to_save = {
                    "mIoU"           : val_metrics["mIoU"],
                    "val_loss"       : val_metrics["val_loss"],
                    "train_loss"     : val_metrics["train_loss"],
                    "fw_iou"         : val_metrics["fw_iou"],
                    "mean_pixel_acc" : val_metrics["mean_pixel_acc"],
                }

                save_checkpoint(
                    model          = self.model,
                    optimizer      = self.optimizer,
                    scheduler      = self.scheduler,
                    epoch          = epoch,
                    metrics        = metrics_to_save,   # ← stripped version
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
                    break   # exits the epoch loop cleanly

            # ── Cleanup ────────────────────────────────────────────────────
            
        if self.tb:
            self.tb.close()
        if self.mlf:
            self.mlf.finish()

        self.logger.info("Training complete.")