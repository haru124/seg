import torch

ckpt = torch.load(
    "outputs/checkpoints/exp3/exp3_adamw_focal_dice_epoch009_loss0.5672_mIoU0.4433.pth",
    map_location="cpu"
)

print(ckpt.keys())
print(ckpt.get("scheduler_state", "NO SCHEDULER STATE"))