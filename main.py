import os, glob, random
import numpy as np
import cv2, nibabel as nib
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------
# Settings
# ---------------------------
DATA_ROOT = r"C:\Users\swaro\Desktop\BRATS\BraTS2020_training_data\content\data"
IMG_SIZE = 128
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------
# Preprocessing (same as PyTorch version)
# ---------------------------
def preprocess_slice_uint8(slice_arr):
    slice_arr = np.nan_to_num(slice_arr)
    p2, p98 = np.percentile(slice_arr, (2, 98))
    slice_clip = np.clip(slice_arr, p2, p98)
    if p98 - p2 > 1e-6:
        slice_scaled = ((slice_clip - p2) / (p98 - p2) * 255.0).astype(np.uint8)
    else:
        slice_scaled = np.zeros_like(slice_clip, dtype=np.uint8)

    g1 = cv2.GaussianBlur(slice_scaled, (3, 3), sigmaX=0.8)
    g2 = cv2.GaussianBlur(slice_scaled, (7, 7), sigmaX=1.5)
    blended = cv2.addWeighted(g1, 0.5, g2, 0.5, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blended)

    clahe_f = clahe_img.astype(np.float32)
    mean, std = clahe_f.mean(), clahe_f.std() + 1e-8
    z = (clahe_f - mean) / std
    return z.astype(np.float32)

def preprocess_volume_slicewise(vol):
    slices = []
    for i in range(vol.shape[2]):
        sl = vol[:, :, i]
        if np.max(sl) == 0: continue
        slices.append(preprocess_slice_uint8(sl))
    return slices

# ---------------------------
# Dataset loader -> tf.data
# ---------------------------
def load_bra_ts_dataset(root_dir, img_size=128, max_slices_per_volume=None):
    samples = []
    patient_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))

    for pdir in tqdm(patient_dirs, desc="Scanning patients"):
        if not os.path.isdir(pdir): continue

        # pick modality
        modality_file = None
        for pref in ("flair","t1ce","t1","t2"):
            found = glob.glob(os.path.join(pdir, f"*{pref}*.nii*"))
            if found: modality_file = found[0]; break
        if modality_file is None:
            found = glob.glob(os.path.join(pdir, "*.nii*"))
            if not found: continue
            modality_file = found[0]

        seg_file_candidates = glob.glob(os.path.join(pdir,"*seg*.nii*"))
        if not seg_file_candidates: continue
        seg_file = seg_file_candidates[0]

        try:
            vol = nib.load(modality_file).get_fdata()
            seg_vol = nib.load(seg_file).get_fdata()
        except Exception as e:
            print(f"Load failed {modality_file}: {e}")
            continue

        proc_slices = preprocess_volume_slicewise(vol)
        mask_slices = [(ms>0).astype(np.uint8) if np.max(ms)>0 else None for ms in seg_vol.transpose(2,0,1)]

        collected = 0
        for idx, preproc in enumerate(proc_slices):
            seg_idx = int(round(idx * (seg_vol.shape[2]/max(1,len(proc_slices)))))
            seg_idx = min(seg_idx, seg_vol.shape[2]-1)
            mask = mask_slices[seg_idx]
            if mask is None:
                if max_slices_per_volume and collected>=max_slices_per_volume:
                    continue
                mask_array = np.zeros_like(preproc,dtype=np.uint8)
            else:
                mask_array = cv2.resize(mask,(preproc.shape[1],preproc.shape[0]),interpolation=cv2.INTER_NEAREST)

            img_resized = cv2.resize(preproc,(img_size,img_size))
            mask_resized = cv2.resize(mask_array,(img_size,img_size),interpolation=cv2.INTER_NEAREST)

            samples.append((img_resized.astype(np.float32),mask_resized.astype(np.uint8)))
            collected+=1
            if max_slices_per_volume and collected>=max_slices_per_volume: break

    imgs = np.expand_dims([s[0] for s in samples], -1)
    masks = np.expand_dims([s[1] for s in samples], -1)
    return imgs, masks

# ---------------------------
# Model: RA-UNet (Keras)
# ---------------------------
def conv_block(x, filters, dropout=0.0):
    res = x
    x = layers.Conv2D(filters,3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(filters,3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    if res.shape[-1]!=filters:
        res = layers.Conv2D(filters,1,padding="same")(res)
    x = layers.Add()([x,res])
    if dropout>0: x = layers.SpatialDropout2D(dropout)(x)
    return x

def recursive_attention_block(x, gating, filters, T=2):
    inter = max(filters//2,8)
    theta = layers.Conv2D(inter,1)(x)
    phi   = layers.Conv2D(inter,1)(x)
    g     = layers.Conv2D(inter,1)(gating)
    att = layers.Multiply()([theta,phi])
    att = layers.Add()([att,g])
    for _ in range(T):
        att = layers.Conv2D(inter,3,padding="same")(att)
        att = layers.BatchNormalization()(att)
        att = layers.ReLU()(att)
    att_out = layers.Conv2D(filters,1,activation="sigmoid")(att)
    return layers.Multiply()([x, layers.Add()([att_out, tf.ones_like(att_out)])])

def build_ra_unet(img_size=128, base_filters=32, dropout=0.0, rab_T=2):
    inputs = keras.Input((img_size,img_size,1))

    e1 = conv_block(inputs,base_filters,dropout); p1 = layers.MaxPool2D()(e1)
    e2 = conv_block(p1,base_filters*2,dropout); p2 = layers.MaxPool2D()(e2)
    e3 = conv_block(p2,base_filters*4,dropout); p3 = layers.MaxPool2D()(e3)
    e4 = conv_block(p3,base_filters*8,dropout); p4 = layers.MaxPool2D()(e4)

    c = conv_block(p4,base_filters*16,dropout)

    u4 = layers.Conv2DTranspose(base_filters*8,2,strides=2)(c)
    e4_att = recursive_attention_block(e4,c,base_filters*8,T=rab_T)
    d4 = conv_block(layers.Concatenate()([u4,e4_att]),base_filters*8,dropout)

    u3 = layers.Conv2DTranspose(base_filters*4,2,strides=2)(d4)
    e3_att = recursive_attention_block(e3,d4,base_filters*4,T=rab_T)
    d3 = conv_block(layers.Concatenate()([u3,e3_att]),base_filters*4,dropout)

    u2 = layers.Conv2DTranspose(base_filters*2,2,strides=2)(d3)
    e2_att = recursive_attention_block(e2,d3,base_filters*2,T=rab_T)
    d2 = conv_block(layers.Concatenate()([u2,e2_att]),base_filters*2,dropout)

    u1 = layers.Conv2DTranspose(base_filters,2,strides=2)(d2)
    e1_att = recursive_attention_block(e1,d2,base_filters,T=rab_T)
    d1 = conv_block(layers.Concatenate()([u1,e1_att]),base_filters,dropout)

    outputs = layers.Conv2D(1,1,activation="sigmoid")(d1)
    return keras.Model(inputs,outputs)

# ---------------------------
# Loss & Metrics
# ---------------------------
def dice_loss(y_true,y_pred,eps=1e-7):
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    num=2*tf.reduce_sum(y_true*y_pred,[1,2,3])
    den=tf.reduce_sum(y_true,[1,2,3])+tf.reduce_sum(y_pred,[1,2,3])+eps
    return 1-tf.reduce_mean(num/den)

def bce_dice_loss(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true,y_pred)+dice_loss(y_true,y_pred)

def dice_coef(y_true,y_pred,th=0.5,eps=1e-7):
    y_pred_bin=tf.cast(y_pred>th,tf.float32)
    num=2*tf.reduce_sum(y_true*y_pred_bin)
    den=tf.reduce_sum(y_true)+tf.reduce_sum(y_pred_bin)+eps
    return num/den

# ---------------------------
# Training + grid search
# ---------------------------
def run_grid_search(imgs,masks,learning_rates=[1e-3],batch_sizes=[8],dropouts=[0.2],optimizers=["adam"],epochs=3):
    results=[]
    best={"dice":-1}
    for lr in learning_rates:
        for bs in batch_sizes:
            for dr in dropouts:
                for opt in optimizers:
                    print(f"CONFIG lr={lr}, bs={bs}, dr={dr}, opt={opt}")
                    model=build_ra_unet(IMG_SIZE,base_filters=32,dropout=dr,rab_T=2)
                    if opt=="adam": optimizer=keras.optimizers.Adam(lr)
                    elif opt=="rmsprop": optimizer=keras.optimizers.RMSprop(lr)
                    else: optimizer=keras.optimizers.SGD(lr,momentum=0.9)
                    model.compile(optimizer,loss=bce_dice_loss,metrics=[dice_coef])
                    hist=model.fit(imgs,masks,batch_size=bs,epochs=epochs,validation_split=0.15,verbose=1)
                    val_dice=max(hist.history["val_dice_coef"])
                    results.append({"lr":lr,"bs":bs,"dropout":dr,"opt":opt,"val_dice":val_dice})
                    if val_dice>best["dice"]:
                        best={"dice":val_dice,"config":{"lr":lr,"bs":bs,"dropout":dr,"opt":opt},"model":model}
    return sorted(results,key=lambda x:x["val_dice"],reverse=True),best

# ---------------------------
# Main
# ---------------------------
if __name__=="__main__":
    imgs,masks=load_bra_ts_dataset(DATA_ROOT,img_size=IMG_SIZE,max_slices_per_volume=10)
    print("Dataset:",imgs.shape,masks.shape)
    results,best=run_grid_search(imgs,masks,
                                 learning_rates=[1e-3,1e-4],
                                 batch_sizes=[8,16],
                                 dropouts=[0.2,0.3],
                                 optimizers=["adam","sgd"],
                                 epochs=3)
    print("Best:",best["config"],"dice",best["dice"])
