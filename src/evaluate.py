import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Ensure project root is in sys.path for reliable module imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.dataset import get_dataloaders
    from src.model import build_model
except ImportError:
    from dataset import get_dataloaders
    from model import build_model


def _resolve_path(path):
    """Resolve relative paths against the HAR-Image Based project root."""
    path = Path(path)
    if not path.is_absolute():
        path = project_root / path
    return path


def evaluate_on_test(model_path, data_root, batch_size=32):
    """
    One-time evaluation of a saved checkpoint on the held-out test set.

    Returns
    -------
    overall_accuracy : float
    report_dict : dict
        sklearn classification_report(..., output_dict=True)
    cm : numpy.ndarray
        Confusion matrix
    class_names : list[str]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = _resolve_path(model_path)

    print("=" * 65)
    print("STAGE 6 — TEST SET EVALUATION (one-time, reporting only)")
    print(
        f"Device Selected  : {device.type.upper()}"
        f" ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    print(f"Checkpoint       : {model_file}")
    print(f"Data Root        : {data_root}")
    print("=" * 65)

    dataloaders = get_dataloaders(data_root, batch_size=batch_size)
    test_loader = dataloaders["test"]
    class_names = dataloaders["class_names"]

    model = build_model(num_classes=12, freeze_backbone=True)
    model.load_state_dict(
        torch.load(model_file, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    overall_accuracy = float(accuracy_score(y_true, y_pred))
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return overall_accuracy, report_dict, cm, class_names


def plot_confusion_matrix(cm, class_names, save_path):
    """Save (and display) a labeled heatmap of the test-set confusion matrix."""
    save_file = _resolve_path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Test Set")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    fig.savefig(save_file, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to: {save_file}")
    plt.show()
    plt.close(fig)


def save_metrics_report(report_dict, overall_accuracy, save_path):
    """Write per-class precision/recall/F1 plus overall accuracy to CSV."""
    save_file = _resolve_path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    fieldnames = ["class_name", "precision", "recall", "f1_score", "support"]

    with open(save_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, metrics in report_dict.items():
            if class_name in skip_keys:
                continue
            writer.writerow({
                "class_name": class_name,
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "f1_score": f"{metrics['f1-score']:.6f}",
                "support": int(metrics["support"]),
            })
        writer.writerow({
            "class_name": "overall_accuracy",
            "precision": f"{overall_accuracy:.6f}",
            "recall": "",
            "f1_score": "",
            "support": "",
        })

    print(f"Metrics CSV saved to: {save_file}")


def _print_classification_report(report_dict, class_names):
    """Print sklearn-style per-class metrics from classification_report's dict."""
    header = f"{'class':<24} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    print("\nPer-class classification report")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name in class_names:
        m = report_dict[name]
        print(
            f"{name:<24} {m['precision']:10.4f} {m['recall']:10.4f} "
            f"{m['f1-score']:10.4f} {int(m['support']):10d}"
        )
    print("-" * len(header))
    for avg_key in ("macro avg", "weighted avg"):
        if avg_key in report_dict:
            m = report_dict[avg_key]
            print(
                f"{avg_key:<24} {m['precision']:10.4f} {m['recall']:10.4f} "
                f"{m['f1-score']:10.4f} {int(m['support']):10d}"
            )


if __name__ == "__main__":
    overall_accuracy, report_dict, cm, class_names = evaluate_on_test(
        "models/best_model.pt",
        "data_split",
    )

    print("\n" + "=" * 65)
    print(f"OVERALL TEST ACCURACY : {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    print("=" * 65)

    _print_classification_report(report_dict, class_names)

    plot_confusion_matrix(cm, class_names, "results/confusion_matrix.png")
    save_metrics_report(report_dict, overall_accuracy, "results/test_metrics.csv")
