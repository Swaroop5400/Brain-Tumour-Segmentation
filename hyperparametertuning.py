import torch
from torch.utils.data import DataLoader, random_split

# ---------------------------
# Training / Validation functions
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            running_loss += loss.item() * imgs.size(0)
            dices.append(compute_dice(preds, masks))
    return running_loss / len(loader.dataset), sum(dices)/len(dices)

# ---------------------------
# Grid Search (Section 4.1)
# ---------------------------
def run_hyperparam_search(dataset,
                          learning_rates=[0.1, 0.01, 0.001, 0.0001],
                          batch_sizes=[8, 16, 32],
                          dropouts=[0.2, 0.3, 0.5],
                          optimizers_list=['adam', 'rmsprop', 'sgd'],
                          epochs=5,
                          device="cuda"):
    """
    Perform grid search over hyperparameters.
    Returns sorted list of results and best config.
    """

    # Train/validation split
    n = len(dataset)
    val_n = int(0.15 * n)
    train_n = n - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    results = []
    best = {"dice": -1.0}

    for lr in learning_rates:
        for bs in batch_sizes:
            for dr in dropouts:
                for opt_name in optimizers_list:

                    print(f"\nConfig: lr={lr}, batch={bs}, dropout={dr}, optim={opt_name}")

                    # DataLoaders
                    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

                    # Model
                    model = RAUNet2D(in_ch=1, out_ch=1, dropout=dr).to(device)

                    # Optimizer
                    if opt_name == "adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    elif opt_name == "rmsprop":
                        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                    # Loss function
                    criterion = bce_dice_loss

                    # Train
                    best_val_dice = -1.0
                    for epoch in range(1, epochs+1):
                        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
                        print(f"Epoch {epoch}/{epochs} - TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValDice {val_dice:.4f}")

                        if val_dice > best_val_dice:
                            best_val_dice = val_dice
                            best_state = model.state_dict()

                    # Record result
                    results.append({
                        "lr": lr,
                        "batch": bs,
                        "dropout": dr,
                        "optimizer": opt_name,
                        "val_dice": best_val_dice
                    })

                    # Update best
                    if best_val_dice > best["dice"]:
                        best = {
                            "dice": best_val_dice,
                            "config": {"lr": lr, "batch": bs, "dropout": dr, "optimizer": opt_name},
                            "state_dict": best_state
                        }

    # Sort by Dice
    results_sorted = sorted(results, key=lambda x: x["val_dice"], reverse=True)

    print("\nTop 5 Results:")
    for r in results_sorted[:5]:
        print(r)

    print("\nBest Config:", best["config"], "with Dice:", best["dice"])

    torch.save(best["state_dict"], "best_ra_unet.pth")
    print("Best model saved as best_ra_unet.pth")

    return results_sorted, best
