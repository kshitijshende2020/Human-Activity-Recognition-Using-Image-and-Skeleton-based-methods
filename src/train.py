import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


# Ensure project root is in sys.path for reliable module imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.dataset import get_dataloaders
    from src.model import build_model, unfreeze_last_blocks
except ImportError:
    from dataset import get_dataloaders
    from model import build_model, unfreeze_last_blocks


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for a single epoch.
    Returns (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = len(dataloader.dataset)

    if total_samples == 0:
        return 0.0, 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation or test dataloader (gradient computation disabled).
    Returns (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = len(dataloader.dataset)

    if total_samples == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def train_model(data_root, num_epochs_phase1=10, num_epochs_phase2=10, batch_size=32,
                checkpoint_dir="models", log_path="results/training_log.csv", resume=False):
    """
    Full 2-phase training pipeline with early stopping, checkpointing, CSV logging, and resume support.
    - Phase 1: Train newly replaced classification head (frozen backbone, Adam lr=1e-3).
    - Phase 2: Unfreeze last residual block (model.layer4[-1]) and fine-tune (Adam lr=1e-5).
    """
    # 1. Reproducibility Seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Resolve paths relative to project root if relative paths given
    chk_dir_path = Path(checkpoint_dir)
    if not chk_dir_path.is_absolute():
        chk_dir_path = project_root / checkpoint_dir

    log_file_path = Path(log_path)
    if not log_file_path.is_absolute():
        log_file_path = project_root / log_path

    # Ensure output folders exist
    chk_dir_path.mkdir(parents=True, exist_ok=True)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    best_model_path = chk_dir_path / "best_model.pt"
    latest_checkpoint_path = chk_dir_path / "latest_checkpoint.pt"

    # 2. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"STAGE 5b — TRAINING & FINE-TUNING PIPELINE")
    print(
        f"Device Selected  : {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Data Root        : {data_root}")
    print(f"Checkpoint Dir   : {chk_dir_path}")
    print(f"Log CSV Path     : {log_file_path}")
    print(f"Resume Enabled   : {resume}")
    print("=" * 65)

    # 3. DataLoaders
    dataloaders = get_dataloaders(data_root, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    criterion = nn.CrossEntropyLoss()

    # Training state trackers
    best_val_acc = 0.0
    patience = 10
    epochs_without_improvement = 0
    log_history = []
    start_epoch = 1
    start_phase = "phase1"
    resumed_checkpoint = None

    # Check for resume option
    if resume:
        if latest_checkpoint_path.exists():
            print(f"\n[RESUME] Loading latest checkpoint from: {latest_checkpoint_path}")
            resumed_checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
            start_epoch = resumed_checkpoint["epoch"] + 1
            start_phase = resumed_checkpoint["phase"]
            best_val_acc = resumed_checkpoint["best_val_acc"]
            epochs_without_improvement = resumed_checkpoint["epochs_without_improvement"]

            # Load existing CSV history if available
            if log_file_path.exists():
                with open(log_file_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if int(row["epoch"]) <= resumed_checkpoint["epoch"]:
                            log_history.append({
                                "epoch": int(row["epoch"]),
                                "phase": row["phase"],
                                "train_loss": float(row["train_loss"]),
                                "train_acc": float(row["train_acc"]),
                                "val_loss": float(row["val_loss"]),
                                "val_acc": float(row["val_acc"])
                            })
            print(
                f"[RESUME] Resuming from Epoch {start_epoch} ({start_phase.upper()}). Best Val Acc so far: {best_val_acc:.4f}")
        else:
            print(
                f"\n[RESUME NOTICE] No latest checkpoint found at {latest_checkpoint_path}. Starting training from scratch.")

    total_phase1_epochs = num_epochs_phase1
    total_phase2_epochs = num_epochs_phase1 + num_epochs_phase2

    # ---------------------------------------------------------
    # PHASE 1: Train Head Only (Frozen Backbone, lr=1e-3)
    # ---------------------------------------------------------
    model = build_model(num_classes=12, freeze_backbone=True)
    model.to(device)
    model = freeze_bn_stats(model)

    trainable_params_p1 = [p for p in model.parameters() if p.requires_grad]
    optimizer_p1 = optim.Adam(trainable_params_p1, lr=1e-3)

    if resumed_checkpoint and start_phase == "phase1":
        model.load_state_dict(resumed_checkpoint["model_state_dict"])
        optimizer_p1.load_state_dict(resumed_checkpoint["optimizer_state_dict"])

    current_epoch = start_epoch

    if start_phase == "phase1" and current_epoch <= total_phase1_epochs:
        print("\n--- PHASE 1: Training Classification Head Only (Adam lr=1e-3) ---")

        while current_epoch <= total_phase1_epochs:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p1, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            log_history.append({
                "epoch": current_epoch,
                "phase": "phase1",
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            print(
                f"Epoch {current_epoch:02d}/{total_phase2_epochs} [Phase 1] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}",
                end="")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f" --> Best Model Saved! (Val Acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                print()

            # Save latest checkpoint after every completed epoch
            latest_chk_data = {
                "epoch": current_epoch,
                "phase": "phase1",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p1.state_dict(),
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement
            }
            torch.save(latest_chk_data, latest_checkpoint_path)

            if epochs_without_improvement >= patience:
                # Early stopping in Phase 1 stops Phase 1 only.
                # Advance current_epoch to total_phase1_epochs + 1 so Phase 2 starts cleanly.
                print(
                    f"\n[EARLY STOPPING] Phase 1 stopped early at epoch {current_epoch} (Patience={patience}). Proceeding to Phase 2.")
                current_epoch += 1
                break

            current_epoch += 1

    # ---------------------------------------------------------
    # PHASE 2: Fine-Tune Last Residual Block + Head (lr=1e-5)
    # ---------------------------------------------------------
    if num_epochs_phase2 > 0:
        # Explicit separation: Resuming mid-Phase-2 vs. Fresh entry into Phase 2
        if resume and start_phase == "phase2" and resumed_checkpoint is not None:
            # --- Path A: Resuming mid-Phase-2 from a saved checkpoint ---
            print("\n--- PHASE 2 (RESUMED): Fine-Tuning Last Residual Block + Head (Adam lr=1e-5) ---")
            model = unfreeze_last_blocks(model, num_blocks=1)
            model.to(device)

            trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
            optimizer_p2 = optim.Adam(trainable_params_p2, lr=1e-5)

            model.load_state_dict(resumed_checkpoint["model_state_dict"])
            optimizer_p2.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        else:
            # --- Path B: Fresh entry into Phase 2 (Phase 1 completed or early-stopped) ---
            print("\n--- PHASE 2: Fine-Tuning Last Residual Block + Head (Adam lr=1e-5) ---")
            # Load best Phase 1 model checkpoint before starting Phase 2 fine-tuning
            if best_model_path.exists():
                model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
                print(f"Loaded best Phase 1 checkpoint ({best_model_path.name}) for Phase 2 initialization.")

            model = unfreeze_last_blocks(model, num_blocks=1)
            model.to(device)

            trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
            optimizer_p2 = optim.Adam(trainable_params_p2, lr=1e-5)

            # Reset patience counter for Phase 2 evaluation
            epochs_without_improvement = 0

            # Ensure current_epoch starts cleanly at Phase 2 boundary (total_phase1_epochs + 1)
            if current_epoch <= total_phase1_epochs:
                current_epoch = total_phase1_epochs + 1

        while current_epoch <= total_phase2_epochs:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p2, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            log_history.append({
                "epoch": current_epoch,
                "phase": "phase2",
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            print(
                f"Epoch {current_epoch:02d}/{total_phase2_epochs} [Phase 2] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}",
                end="")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f" --> Best Model Saved! (Val Acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                print()

            # Save latest checkpoint after every completed epoch
            latest_chk_data = {
                "epoch": current_epoch,
                "phase": "phase2",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p2.state_dict(),
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement
            }
            torch.save(latest_chk_data, latest_checkpoint_path)

            if epochs_without_improvement >= patience:
                print(f"\n[EARLY STOPPING] Phase 2 stopped early at epoch {current_epoch} (Patience={patience}).")
                break

            current_epoch += 1

    # 4. Save Training History Log to CSV
    fieldnames = ["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc"]
    with open(log_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_history)

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 65)
    print(f"Best Validation Accuracy : {best_val_acc:.4f}")
    print(f"Best Checkpoint Saved    : {best_model_path}")
    print(f"Latest Checkpoint Saved  : {latest_checkpoint_path}")
    print(f"Training Log CSV Saved   : {log_file_path}")
    print("=" * 65)

    # Load best checkpoint weights before returning
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:
        print(
            f"\n[WARNING] best_model.pt was never written during training (val accuracy never exceeded 0.0). Retaining current in-memory model state.")

    return model, log_history


if __name__ == "__main__":
    print("Stage 5b training module (src/train.py) loaded successfully and ready.")