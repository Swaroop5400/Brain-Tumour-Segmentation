import numpy as np
import matplotlib.pyplot as plt

def plot_undersegmentation_heatmap(image, pred_mask, gt_mask, alpha=0.5, cmap="Reds"):
    """
    image: 2D MRI slice (H,W)
    pred_mask: binary predicted mask (H,W)
    gt_mask: binary ground truth mask (H,W)
    """
    # False Negatives = missed tumor pixels
    underseg = np.logical_and(gt_mask == 1, pred_mask == 0).astype(np.float32)

    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap="gray")
    plt.imshow(underseg, cmap=cmap, alpha=alpha)  # heatmap overlay
    plt.title("Under-segmentation Heatmap")
    plt.axis("off")
    plt.show()
