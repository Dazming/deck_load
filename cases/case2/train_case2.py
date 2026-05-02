"""
//AI辅助生成: Qwen3.5， 2026-4-20
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.data_pipeline import prepare_data
from shared.model_arch import AMFBiGRU


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return AMFBiGRU(
        disp_input_dim=config.DISP_FEATURES,
        acc_input_dim=config.ACC_FEATURES,
        hidden_dim=config.BIGRU_HIDDEN,
        fc1_dim=config.FC1_DIM,
        fc2_dim=config.FC2_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    )


def train():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, _ = prepare_data(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME))
        else:
            patience_counter += 1

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:4d} | Train: {train_loss:.6e} | Val: {val_loss:.6e} | "
            f"Best: {best_val_loss:.6e} (ep {best_epoch}) | LR: {current_lr:.2e} | "
            f"Patience: {patience_counter}/{config.EARLY_STOP_PATIENCE} | {elapsed:.0f}s"
        )

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s. Best val loss: {best_val_loss:.6e} at epoch {best_epoch}")
    print(f"Model saved to {os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME)}")


if __name__ == "__main__":
    train()

