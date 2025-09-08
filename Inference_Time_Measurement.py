
import time
import numpy as np
import torch

def measure_inference_time(model, dataloader, device=torch.device("cpu"), n_warmup=5, max_batches=None):
    """
    model: PyTorch model
    dataloader: yields (img_tensor, mask) where img_tensor shape [B,1,H,W]
    device: CPU or GPU (cuda)
    n_warmup: skip first n_warmup passes to avoid cold-start overhead
    max_batches: optional to limit measurement
    Returns: dict with per-slice mean/std (seconds) and total samples measured
    """
    model.to(device)
    model.eval()
    times = []
    total_samples = 0
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            imgs = imgs.to(device)
            # warmup
            if i < n_warmup:
                _ = model(imgs)
                continue
            t0 = time.perf_counter()
            _ = model(imgs)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            # per-slice time
            per_slice = elapsed / imgs.size(0)
            times.append(per_slice)
            total_samples += imgs.size(0)
    times = np.array(times)
    return {
        "per_slice_mean_s": float(times.mean()) if len(times)>0 else 0.0,
        "per_slice_std_s": float(times.std(ddof=1)) if len(times)>1 else 0.0,
        "batches_measured": len(times),
        "samples_measured": total_samples
    }

# Example of extrapolating to per-scan:
def estimate_scan_time(per_slice_mean_s, slices_per_scan):
    """
    per_slice_mean_s: mean seconds per 2D slice
    slices_per_scan: number of axial slices in a full 3D scan
    """
    total = per_slice_mean_s * slices_per_scan
    return total

""" to use the code , tun the below snippet:
# assume model, val_loader defined and device set
timing = measure_inference_time(model, val_loader, device=DEVICE, n_warmup=3, max_batches=50)
print("Per-slice: {:.4f} ± {:.4f} s".format(timing["per_slice_mean_s"], timing["per_slice_std_s"]))
# if typical full scan has 20 axial slices:
scan_seconds = estimate_scan_time(timing["per_slice_mean_s"], slices_per_scan=20)
print(f"Estimated per-scan time (20 slices): {scan_seconds:.2f} s")
"""



"""
