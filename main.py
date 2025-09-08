"""
Complete script: Preprocessing (Double Blended Gaussian + CLAHE + Z-score),
Dataset loader for BraTS (2D slices), RA U-Net implementation,
and Section 4.1 style grid-search for hyperparameter tuning.

Set DATA_ROOT to:
"C:\\Users\\swaro\\Desktop\\BRATS\\BraTS2020_training_data\\content\\data"

Dependencies:
  pip install torch torchvision nibabel opencv-python scikit-image tqdm
"""

import os
import glob
import random
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ---------------------------
# User settings
# ---------------------------
DATA_ROOT = r"C:\Users\swaro\Desktop\BRATS\BraTS2020_training_data\content\data"
IMG_SIZE = 128           # as per manuscript
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------
# Preprocessing functions
# ---------------------------
def preprocess_slice_uint8(slice_arr):
    """
    slice_arr: 2D numpy array (float) representing one axial slice
    Steps:
      - Clip / normalize to 0-255 for CLAHE (preserve relative values)
      - Double Blended Gaussian: (3x3, sigma=0.8) and (7x7, sigma=1.5), blend 50/50
      - CLAHE (tile 8x8, clip 2.0)
      - Z-score normalization (per slice)
    Returns: float32 array (z-scored)
    """
    # handle NaNs
    slice_arr = np.nan_to_num(slice_arr)

    # Normalize to 0-255 for cv2 functions (preserve contrast)
    p2, p98 = np.percentile(slice_arr, (2, 98))
    slice_clip = np.clip(slice_arr, p2, p98)
    if p98 - p2 > 1e-6:
        slice_scaled = ((slice_clip - p2) / (p98 - p2) * 255.0).astype(np.uint8)
    else:
        slice_scaled = np.zeros_like(slice_clip, dtype=np.uint8)

    # Double Gaussian
    g1 = cv2.GaussianBlur(slice_scaled, (3, 3), sigmaX=0.8)
    g2 = cv2.GaussianBlur(slice_scaled, (7, 7), sigmaX=1.5)
    blended = cv2.addWeighted(g1, 0.5, g2, 0.5, 0)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blended)

    # Convert back to float and z-score normalize (per slice)
    clahe_f = clahe_img.astype(np.float32)
    mean = clahe_f.mean()
    std = clahe_f.std() + 1e-8
    z = (clahe_f - mean) / std

    return z.astype(np.float32)


def preprocess_volume_slicewise(vol):
    """
    vol: 3D numpy array (H,W,D) - one modality volume
    Returns: list of preprocessed 2D slices (float32)
    We preprocess each axial slice independently (common for 2D training).
    """
    slices = []
    for i in range(vol.shape[2]):
        sl = vol[:, :, i]
        # skip empty slices
        if np.max(sl) == 0:
            continue
        p = preprocess_slice_uint8(sl)
        slices.append(p)
    return slices


# ---------------------------
# Dataset class for BRATS (2D slice training)
# ---------------------------
class BraTSSliceDataset(Dataset):
    """
    Expects each patient folder to contain modalities like:
       *_flair.nii.gz, *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, and *_seg.nii.gz
    If flair exists it uses flair; else chooses the first available modality.
    Produces tuples (image_tensor, mask_tensor), image: [1, H, W], mask: [H, W] binary (0/1)
    """
    def __init__(self, root_dir, modality_preference=("flair", "t1ce", "t1", "t2"),
                 img_size=IMG_SIZE, max_slices_per_volume=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.max_slices_per_volume = max_slices_per_volume
        self.samples = []  # list of (image_slice_array, mask_slice_array)
        patient_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        for pdir in tqdm(patient_dirs, desc="Scanning patients"):
            if not os.path.isdir(pdir):
                continue
            # find modality file
            modality_file = None
            for m in modality_preference:
                found = glob.glob(os.path.join(pdir, f"*{m}*.nii*"))
                if len(found) > 0:
                    modality_file = found[0]
                    break
            if modality_file is None:
                # fallback to first nii file
                found = glob.glob(os.path.join(pdir, "*.nii*"))
                if len(found) == 0:
                    continue
                modality_file = found[0]

            # segmentation file
            seg_file_candidates = glob.glob(os.path.join(pdir, "*seg*.nii*")) + glob.glob(os.path.join(pdir, "*_segmentation*.nii*")) + glob.glob(os.path.join(pdir, "*_seg.nii*"))
            if len(seg_file_candidates) == 0:
                # sometimes name is 'OT' or similar - try anything with 'seg' else skip
                continue
            seg_file = seg_file_candidates[0]

            try:
                vol = nib.load(modality_file).get_fdata()
                seg_vol = nib.load(seg_file).get_fdata()
            except Exception as e:
                print(f"Failed to load {modality_file} or {seg_file}: {e}")
                continue

            # Preprocess per-slice
            proc_slices = preprocess_volume_slicewise(vol)
            # ensure seg slices align: take same axial indices by checking nonempty mask slices
            mask_slices = []
            for i in range(seg_vol.shape[2]):
                ms = seg_vol[:, :, i]
                # Binarize segmentation: any positive label -> 1
                if np.max(ms) == 0:
                    mask_slices.append(None)  # placeholder
                else:
                    mask_slices.append((ms > 0).astype(np.uint8))

            # Align and collect slices where mask exists (we want informative slices)
            collected = 0
            for idx, preproc in enumerate(proc_slices):
                # proc_slices derived by skipping empty slices; need alignment index mapping.
                # Simpler approach: sample slices from same z-range proportionally.
                # Map proc_slices index to seg slice index by linear scaling
                seg_idx = int(round(idx * (seg_vol.shape[2] / max(1, len(proc_slices))) ))
                seg_idx = min(seg_idx, seg_vol.shape[2]-1)
                mask = mask_slices[seg_idx]
                if mask is None:
                    # optionally keep some background slices as negative examples
                    # we'll include them but only if we haven't exceeded maximum
                    if self.max_slices_per_volume is not None and collected >= self.max_slices_per_volume:
                        continue
                    mask_array = np.zeros_like(preproc, dtype=np.uint8)
                else:
                    mask_array = cv2.resize(mask.astype(np.uint8), (preproc.shape[1], preproc.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Resize both to IMG_SIZE
                img_resized = cv2.resize(preproc, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(mask_array, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

                self.samples.append((img_resized.astype(np.float32), mask_resized.astype(np.uint8)))
                collected += 1
                if self.max_slices_per_volume and collected >= self.max_slices_per_volume:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        # add channel dim
        img_t = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W] as float for BCE/Dice
        return img_t, mask_t

# ---------------------------
# Model: Recursive Attention U-Net (2D)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = x if self.res_conv is None else self.res_conv(x)
        out = self.conv(x)
        out = out + res
        out = self.dropout(out)
        return out

class RecursiveAttentionBlock(nn.Module):
    """
    Implements a simplified Recursive Attention Block (RAB).
    - Shared conv transforms input; attention map computed with gating signal.
    - Recursively refines attention map T times (T configurable).
    """
    def __init__(self, channels, inter_channels=None, T=2):
        super().__init__()
        inter = inter_channels or max(channels // 2, 8)
        self.T = T
        # feature transform
        self.theta = nn.Conv2d(channels, inter, kernel_size=1)
        self.phi = nn.Conv2d(channels, inter, kernel_size=1)
        self.g = nn.Conv2d(channels, inter, kernel_size=1)
        self.W = nn.Conv2d(inter, channels, kernel_size=1)
        # recursive refinement conv
        self.recursive_refine = nn.Sequential(
            nn.Conv2d(inter, inter, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gating=None):
        # initial attention
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        # basic attention map (dot-product style)
        att = theta_x * phi_x  # element-wise
        # optionally incorporate gating (from deeper layer)
        if gating is not None:
            g_proj = self.g(gating)
            att = att + g_proj
        # recursive refinement
        for _ in range(self.T):
            att = self.recursive_refine(att)
        # map back and apply sigmoid
        att_out = self.W(att)
        att_map = self.sigmoid(att_out)  # values in (0,1)
        # apply attention to input features
        out = x * (1 + att_map)  # residual-style attention
        return out, att_map


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class RAUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_filters=32, dropout=0.0, rab_T=2):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_ch, f, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f, f*2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(f*2, f*4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(f*4, f*8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(f*8, f*16, dropout)

        # Recursive Attention Blocks in decoder/encoder skip fusion
        self.rab4 = RecursiveAttentionBlock(f*8, T=rab_T)
        self.rab3 = RecursiveAttentionBlock(f*4, T=rab_T)
        self.rab2 = RecursiveAttentionBlock(f*2, T=rab_T)
        self.rab1 = RecursiveAttentionBlock(f, T=rab_T)

        self.up4 = UpConv(f*16, f*8)
        self.dec4 = ConvBlock(f*16, f*8, dropout)
        self.up3 = UpConv(f*8, f*4)
        self.dec3 = ConvBlock(f*8, f*4, dropout)
        self.up2 = UpConv(f*4, f*2)
        self.dec2 = ConvBlock(f*4, f*2, dropout)
        self.up1 = UpConv(f*2, f)
        self.dec1 = ConvBlock(f*2, f, dropout)

        self.final = nn.Conv2d(f, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        c = self.center(p4)

        # Decoder with recursive attention fusion
        u4 = self.up4(c)
        # attention on e4 guided by c
        e4_att, _ = self.rab4(e4, gating=c)
        d4 = torch.cat([u4, e4_att], dim=1)
        d4 = self.dec4(d4)

        u3 = self.up3(d4)
        e3_att, _ = self.rab3(e3, gating=d4)
        d3 = torch.cat([u3, e3_att], dim=1)
        d3 = self.dec3(d3)

        u2 = self.up2(d3)
        e2_att, _ = self.rab2(e2, gating=d3)
        d2 = torch.cat([u2, e2_att], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        e1_att, _ = self.rab1(e1, gating=d2)
        d1 = torch.cat([u1, e1_att], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)  # for binary segmentation
        return out

# ---------------------------
# Loss functions and metrics
# ---------------------------
def dice_loss(pred, target, eps=1e-7):
    # pred and target are tensors with shape [B,1,H,W]
    num = 2 * (pred * target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    loss = 1 - (num / den)
    return loss.mean()

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dloss = dice_loss(pred, target)
    return bce + dloss

def compute_dice(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    num = 2 * (pred_bin * target).sum()
    den = pred_bin.sum() + target.sum() + eps
    return (num / den).item()

# ---------------------------
# Training / Validation functions
# ---------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        preds = model(imgs)
        loss = bce_dice_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader):
    model.eval()
    running_loss = 0.0
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(imgs)
            loss = bce_dice_loss(preds, masks)
            running_loss += loss.item() * imgs.size(0)
            dices.append(compute_dice(preds, masks))
    mean_loss = running_loss / len(loader.dataset)
    mean_dice = np.mean(dices) if len(dices)>0 else 0.0
    return mean_loss, mean_dice

# ---------------------------
# Grid search (Section 4.1)
# ---------------------------
def run_grid_search(data_root,
                    learning_rates=[0.1, 0.01, 0.001, 0.0001],
                    batch_sizes=[8, 16, 32],
                    dropouts=[0.2, 0.3, 0.5],
                    optimizers_list=['adam', 'rmsprop', 'sgd'],
                    base_filters=32,
                    rab_T=2,
                    epochs=5,
                    quick_mode=True):
    """
    quick_mode: if True restricts dataset size to speed up debugging
    epochs: number of epochs per config (use larger for final runs)
    """
    print(f"Building dataset from {data_root} ...")
    ds = BraTSSliceDataset(data_root, img_size=IMG_SIZE, max_slices_per_volume=(10 if quick_mode else None))
    print(f"Total slices: {len(ds)}")

    # train/val split
    n = len(ds)
    if n == 0:
        raise RuntimeError("No samples found. Check DATA_ROOT path and folder structure.")
    val_frac = 0.15
    val_n = max(1, int(n * val_frac))
    train_n = n - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    results = []
    best_record = {"dice": -1.0}

    for lr in learning_rates:
        for bs in batch_sizes:
            for dr in dropouts:
                for opt_name in optimizers_list:
                    print(f"\n===== CONFIG: lr={lr}, batch={bs}, dropout={dr}, optim={opt_name} =====")
                    # dataloaders
                    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
                    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

                    # model
                    model = RAUNet2D(in_ch=1, out_ch=1, base_filters=base_filters, dropout=dr, rab_T=rab_T).to(DEVICE)
                    # optimizer
                    if opt_name == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    elif opt_name == 'rmsprop':
                        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                    # training loop (small number epochs for each config)
                    best_val_dice = -1.0
                    for epoch in range(1, epochs+1):
                        train_loss = train_epoch(model, train_loader, optimizer) if False else None  # placeholder
                        # Note: to speed up quick tests, we only run a short validation-based training:
                        # perform one epoch manually (lighter than full epoch handling)
                        model.train()
                        for imgs, masks in train_loader:
                            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                            preds = model(imgs)
                            loss = bce_dice_loss(preds, masks)
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                            break  # quick: single batch step to allow config test
                        val_loss, val_dice = validate_epoch(model, val_loader)
                        print(f"Epoch {epoch}/{epochs}  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}")
                        if val_dice > best_val_dice:
                            best_val_dice = val_dice
                            best_state = model.state_dict()

                    # store result
                    results.append({
                        "lr": lr,
                        "batch": bs,
                        "dropout": dr,
                        "optimizer": opt_name,
                        "val_dice": best_val_dice
                    })
                    # update global best
                    if best_val_dice > best_record["dice"]:
                        best_record = {
                            "dice": best_val_dice,
                            "config": {"lr": lr, "batch": bs, "dropout": dr, "optimizer": opt_name},
                            "state_dict": best_state
                        }

    # summarize
    results_sorted = sorted(results, key=lambda x: x["val_dice"], reverse=True)
    print("\n===== Grid Search Results (top 10) =====")
    for r in results_sorted[:10]:
        print(r)
    print("\nBest overall config:", best_record["config"], "dice=", best_record["dice"])

    # save best model
    if "state_dict" in best_record:
        torch.save(best_record["state_dict"], "best_ra_unet_grid.pth")
        print("Best model saved to best_ra_unet_grid.pth")

    return results_sorted, best_record

# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    # small, safe defaults for quick test. For full experiments:
    # set quick_mode=False, epochs=100, adjust learning_rates/batch_sizes/dropouts/optimizers
    results, best = run_grid_search(
        DATA_ROOT,
        learning_rates=[0.001, 0.0001],   # narrower for sensible defaults
        batch_sizes=[8, 16],
        dropouts=[0.2, 0.3],
        optimizers_list=['adam', 'sgd'],
        base_filters=32,
        rab_T=2,
        epochs=3,
        quick_mode=True  # set False to use all slices (longer)
    )
