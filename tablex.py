import pandas as pd

# Suppose you have lists/arrays for each test subject:
# ra_dice_list, baseline_dice_list, ra_iou_list, baseline_iou_list
# and per-subject slice counts slices_per_subject (for scan-level timing estimation)

# Compute CIs (t-based) for RA model:
ra_dice_mean, ra_dice_lo, ra_dice_hi = ci_t_based(ra_dice_list)
ra_iou_mean, ra_iou_lo, ra_iou_hi = ci_t_based(ra_iou_list)

# paired p-values vs baseline
t_dice, pval_dice = paired_ttest_pvalue(np.array(ra_dice_list), np.array(baseline_dice_list))
t_iou, pval_iou = paired_ttest_pvalue(np.array(ra_iou_list), np.array(baseline_iou_list))

# inference timing measurement (see above)
timing = measure_inference_time(model, val_loader, device=DEVICE, n_warmup=5, max_batches=100)
per_slice_mean = timing["per_slice_mean_s"]
per_slice_std = timing["per_slice_std_s"]

# estimate per-scan mean using average slices per scan
avg_slices = int(np.mean(slices_per_subject))  # e.g., 20
est_scan_time = estimate_scan_time(per_slice_mean, avg_slices)

# assemble results table
rows = [{
    "Model": "RA-U-Net",
    "Dice_mean": ra_dice_mean,
    "Dice_CI_low": ra_dice_lo,
    "Dice_CI_high": ra_dice_hi,
    "IoU_mean": ra_iou_mean,
    "IoU_CI_low": ra_iou_lo,
    "IoU_CI_high": ra_iou_hi,
    "Dice_vs_baseline_p": pval_dice,
    "IoU_vs_baseline_p": pval_iou,
    "Per_slice_time_s": per_slice_mean,
    "Per_scan_time_s": est_scan_time
}]

df_results = pd.DataFrame(rows)
df_results.to_csv("ra_unet_eval_summary.csv", index=False)
print(df_results.to_string(index=False))
