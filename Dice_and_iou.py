import numpy as np
import torch

def dice_score_np(pred_bin, gt_bin, eps=1e-7):
    # pred_bin, gt_bin: numpy arrays of same shape, binary {0,1}
    num = 2.0 * np.sum(pred_bin * gt_bin)
    den = np.sum(pred_bin) + np.sum(gt_bin) + eps
    return num / den

def iou_score_np(pred_bin, gt_bin, eps=1e-7):
    num = np.sum(pred_bin * gt_bin)
    den = np.sum((pred_bin + gt_bin) > 0) + eps
    return num / den

# PyTorch versions (for batched tensors)
def dice_score_torch(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    num = 2.0 * (pred_bin * target).sum(dim=(1,2,3))
    den = pred_bin.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (num / den).cpu().numpy()  # returns per-batch array

def iou_score_torch(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    num = (pred_bin * target).sum(dim=(1,2,3))
    den = ((pred_bin + target) > 0).sum(dim=(1,2,3)) + eps
    return (num / den).cpu().numpy()
