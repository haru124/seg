# src/seg/profiler/profiler_utils.py
# NOTE: Run profiling LAST — after training loop is stable
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from pathlib import Path


def profile_model(model, dataloader, device, log_dir="outputs/profiler", num_batches=5):
    """
    Profiles forward pass + backward pass for num_batches.
    View results: tensorboard --logdir outputs/profiler
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    model.train().to(device)

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=num_batches, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i, (images, targets) in enumerate(dataloader):
            if i >= num_batches + 2:
                break
            images = images.to(device)
            targets = targets.to(device).long()

            with record_function("forward"):
                output = model(images)

            loss = output["out"].mean() if isinstance(output, dict) else output.mean()

            with record_function("backward"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"[Profiler] Trace saved → {log_dir}")