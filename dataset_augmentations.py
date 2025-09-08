def __getitem__(self, idx):
    img, mask = self.samples[idx]

    # Albumentations expects dict
    augmented = train_transform(image=img, mask=mask)
    img_aug, mask_aug = augmented['image'], augmented['mask']

    # Z-score normalize
    img_aug = zscore_normalize(img_aug)

    # Convert to torch tensors
    img_t = torch.from_numpy(img_aug).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask_aug).unsqueeze(0).float()

    return img_t, mask_t
