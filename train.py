import os
import time

import numpy as np
import torch
import torch.nn as nn

import config
from data_loader import prepare_data
from model import AMFBiGRU


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _, target_scaler = prepare_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = AMFBiGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config.LR_SCHEDULER_PATIENCE,
        factor=config.LR_SCHEDULER_FACTOR,
        min_lr=config.LR_MIN,
    )
    criterion = nn.MSELoss()

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    t_start = time.time()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x_disp, x_acc, y in train_loader:
            x_disp, x_acc, y = x_disp.to(device), x_acc.to(device), y.to(device)
            pred = model(x_disp, x_acc)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_disp, x_acc, y in val_loader:
                x_disp, x_acc, y = x_disp.to(device), x_acc.to(device), y.to(device)
                pred = model(x_disp, x_acc)
                val_loss += criterion(pred, y).item() * y.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, "best_model.pth"))
        else:
            patience_counter += 1

        if True:
            elapsed = time.time() - t_start
            print(
                f"Epoch {epoch:4d} | Train: {train_loss:.6e} | "
                f"Val: {val_loss:.6e} | Best: {best_val_loss:.6e} (ep {best_epoch}) | "
                f"LR: {current_lr:.2e} | Patience: {patience_counter}/{config.EARLY_STOP_PATIENCE} | "
                f"{elapsed:.0f}s"
            )

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s. Best val loss: {best_val_loss:.6e} at epoch {best_epoch}")
    print(f"Model saved to {os.path.join(config.SAVE_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    train()
