# main.py
import argparse
import torch
import torch.optim as optim

from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH
from src.seg.utils.common import set_seed, get_device, count_parameters
from src.seg.datasets.dataloader import build_dataloader
from src.seg.losses.losses import build_loss
from src.seg.training.trainer import Trainer


def build_model(cfg):
    # Import here so you can swap segmt.py contents from the reference repo
    from src.seg.models.deeplabv3_plus import get_segmentation_model
    model = get_segmentation_model(
        model="deeplabv3_plus",
        dataset="cityscapes",
        backbone=cfg.model.backbone,
        aux=cfg.training.aux_loss,
        pretrained_base=cfg.model.pretrained_backbone,
    )
    return model


def build_optimizer(model, cfg):
    opt_name = getattr(cfg.training, "optimizer", "sgd").lower()
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg, num_iters_per_epoch: int):
    if cfg.training.lr_scheduler == "poly":
        total_iters = cfg.training.epochs * num_iters_per_epoch
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda i: (1 - i / total_iters) ** 0.9
        )
    elif cfg.training.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    elif cfg.training.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return None


def main(args):
    cfg = get_config(CONFIG_PATH, args.exp_config)
    set_seed(42)
    device = get_device()

    train_loader = build_dataloader(cfg.data, split="train")
    val_loader = build_dataloader(cfg.data, split="val")

    model = build_model(cfg)
    print(f"[Model] {count_parameters(model)}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    
    loss_cfg = cfg["loss"]   # pass raw dict from yaml
    loss_fn = build_loss( loss_cfg["type"], ignore_index=cfg.data.ignore_index,
            **{k: v for k, v in loss_cfg.items() if k != "type"}
                        )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default=None,
                        help="Path to experiment yaml, e.g. config/experiments/exp_01.yaml")
    args = parser.parse_args()
    main(args)