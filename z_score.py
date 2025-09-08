import numpy as np
import torch

def zscore_normalize(image, eps=1e-8):
    """
    Z-score normalization for a single image (numpy or torch).
    image: 2D numpy array or torch tensor
    Returns: normalized image
    """
    if isinstance(image, np.ndarray):
        mean, std = image.mean(), image.std()
        return (image - mean) / (std + eps)
    elif torch.is_tensor(image):
        mean, std = image.mean(), image.std()
        return (image - mean) / (std + eps)
    else:
        raise TypeError("Input must be numpy array or torch tensor")

"""to use this add in preprocessing slide:
img_resized = cv2.resize(preproc, (self.img_size, self.img_size))
mask_resized = cv2.resize(mask_array, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

# Z-score normalize
img_resized = zscore_normalize(img_resized)
"""
