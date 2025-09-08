import time
import torch
import numpy as np

def measure_runtime(model, dataloader, device, warmup=5, max_batches=None):
    """
    Measure runtime efficiency of a segmentation model.
    Returns per-slice time, throughput, memory usage.
    """
    model.to(device)
    model.eval()

    times = []
    n_samples = 0

    torch.backends.cudnn.benchmark = True  # optimize convs

    with torch.inference_mode():
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            if i < warmup:
                _ = model(imgs)  # warmup
                continue
            start = time.perf_counter()
            _ = model(imgs)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.perf_counter()
            elapsed = end - start
            per_sample = elapsed / imgs.size(0)
            times.append(per_sample)
            n_samples += imgs.size(0)

            if max_batches and i >= max_batches:
                break

    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    throughput = 1.0 / mean_time

    mem_alloc = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0

    return {
        "per_slice_time_s": mean_time,
        "per_slice_std_s": std_time,
        "throughput_slices_per_s": throughput,
        "samples_measured": n_samples,
        "gpu_mem_MB": mem_alloc
    }
