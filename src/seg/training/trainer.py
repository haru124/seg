# src/seg/training/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from src.seg.evaluation.metrics import SegmentationMetrics
from src.seg.utils.checkpoint import save_checkpoint, load_checkpoint
from src.seg.utils.common import setup_logger
from src.seg.tracking.mlflow_logger import MLflowLogger
from src.seg.tracking.tensorboard_logger import TensorboardLogger
from src.seg.entity.config_entity import ExperimentConfig


class Trainer:
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
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.scaler = GradScaler(enabled=cfg.training.amp)

        self.metrics = SegmentationMetrics(
            num_classes=cfg.data.num_classes,
            ignore_index=cfg.data.ignore_index
        )
        self.logger = setup_logger(
            "trainer", "outputs/logs", cfg.experiment_id
        )
        self.tb = TensorboardLogger(cfg.tracking.tb_log_dir, cfg.experiment_id) \
            if cfg.tracking.tb_enabled else None
        self.mlf = MLflowLogger(
            cfg.tracking.mlflow_uri,
            cfg.tracking.mlflow_experiment,
            run_name=cfg.experiment_id
        ) if cfg.tracking.mlflow_enabled else None

        # Resume if needed
        self.start_epoch = 1
        if cfg.checkpoint.resume:
            state = load_checkpoint(cfg.checkpoint.resume, model, optimizer, scheduler, str(device))
            self.start_epoch = state["epoch"] + 1

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        accum = self.cfg.training.accumulation_steps

        self.optimizer.zero_grad()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f"[Epoch {epoch}] Train")

        for i, (images, targets) in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            with autocast(enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets) / accum

            self.scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(self.train_loader):
                if self.cfg.training.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"[Epoch {epoch}] Val ")
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            with autocast(enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            preds = outputs["out"] if isinstance(outputs, dict) else outputs
            preds = preds.argmax(dim=1)
            self.metrics.update(preds, targets)
            total_loss += loss.item()

        result = self.metrics.compute()
        result["val_loss"] = total_loss / len(self.val_loader)
        return result

    def train(self):
        cfg = self.cfg

        if self.mlf:
            self.mlf.log_params({
                "backbone": cfg.model.backbone,
                "output_stride": cfg.model.output_stride,
                "lr": cfg.training.lr,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.data.batch_size,
                "amp": cfg.training.amp,
            })

        for epoch in range(self.start_epoch, cfg.training.epochs + 1):
            train_loss = self._train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} | train_loss={train_loss:.4f}")

            val_metrics = self._val_epoch(epoch)
            val_metrics["train_loss"] = train_loss
            self.logger.info(
                f"Epoch {epoch} | val_loss={val_metrics['val_loss']:.4f} "
                f"| mIoU={val_metrics['mIoU']:.4f} | pixAcc={val_metrics['pixAcc']:.4f}"
            )

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Logging
            step = epoch
            if self.tb:
                self.tb.log_scalars({"train_loss": train_loss}, step, prefix="Loss")
                self.tb.log_scalars(val_metrics, step, prefix="Val")
            if self.mlf:
                self.mlf.log_metrics({"train_loss": train_loss}, step=step)
                self.mlf.log_metrics({k: v for k, v in val_metrics.items()
                                       if isinstance(v, float)}, step=step)

            # Checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_metrics,
                exp_id=cfg.experiment_id,
                checkpoint_dir=cfg.checkpoint.dir,
                top_k=cfg.checkpoint.save_top_k,
            )

        if self.tb:
            self.tb.close()
        if self.mlf:
            self.mlf.finish()
        self.logger.info("Training complete.")
