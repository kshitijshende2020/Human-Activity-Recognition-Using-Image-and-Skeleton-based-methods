import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def resolve_data_root(data_root):
    """
    Resolves data_root path flexibly whether script is executed from project root,
    src/ directory, or any other working directory.
    """
    path = Path(data_root)
    if path.exists():
        return path
    
    project_root = Path(__file__).resolve().parent.parent
    rel_path = project_root / data_root
    if rel_path.exists():
        return rel_path

    return path

def get_class_names(data_root):
    """
    Returns a sorted list of action class names derived from subfolders inside {data_root}/train/.
    This ensures consistent class-to-index mapping across all splits and dataset versions.
    """
    data_root_path = resolve_data_root(data_root)
    train_dir = data_root_path / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found at: {train_dir}")
        
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return class_names

class TennisActionDataset(Dataset):
    """
    PyTorch Dataset for Tennis Action Recognition.
    Loads images from {root_dir}/{split}/<action>/*.jpg using PIL.
    """
    def __init__(self, root_dir, split, transform=None):
        super().__init__()
        self.root_dir = resolve_data_root(root_dir)
        self.split = split
        self.transform = transform

        # Always derive class list from the 'train' split directory to maintain index consistency
        self.class_names = get_class_names(self.root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.samples = []
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for cls_name in self.class_names:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue

            cls_idx = self.class_to_idx[cls_name]
            # Match .jpg and .jpeg image files
            for img_path in cls_dir.glob("*.[jJ][pP][gG]"):
                self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_idx

def get_transforms(input_size=224):
    """
    Returns data augmentation & normalization transforms for train and validation/test splits.
    Uses standard ImageNet normalization statistics.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return {
        "train": train_transform,
        "val_test": val_test_transform
    }

def get_dataloaders(root_dir, batch_size=32, input_size=224, num_workers=2):
    """
    Builds and returns DataLoaders for 'train', 'val', and 'test' splits,
    along with the list of action class names.
    """
    tfms = get_transforms(input_size=input_size)
    class_names = get_class_names(root_dir)

    train_dataset = TennisActionDataset(root_dir=root_dir, split="train", transform=tfms["train"])
    val_dataset = TennisActionDataset(root_dir=root_dir, split="val", transform=tfms["val_test"])
    test_dataset = TennisActionDataset(root_dir=root_dir, split="test", transform=tfms["val_test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": class_names
    }

if __name__ == "__main__":
    print("=" * 65)
    print("STAGE 4 — PYTORCH DATASET & DATALOADER SANITY CHECK")
    print("=" * 65)

    # 1. Sanity check for 1-frame dataset ("data_split")
    print("\n--- Testing 1-Frame Dataset ('data_split') ---")
    dataloaders_1frame = get_dataloaders("data_split", batch_size=32, num_workers=0)
    class_names_1f = dataloaders_1frame["class_names"]

    print(f"Number of Action Classes : {len(class_names_1f)}")
    print(f"Class Names              : {class_names_1f}")
    print(f"Train Dataset Size       : {len(dataloaders_1frame['train'].dataset)} images")
    print(f"Val Dataset Size         : {len(dataloaders_1frame['val'].dataset)} images")
    print(f"Test Dataset Size        : {len(dataloaders_1frame['test'].dataset)} images")

    # Fetch one sample batch from train loader
    sample_images_1f, sample_labels_1f = next(iter(dataloaders_1frame["train"]))
    print(f"Sample Batch Images Shape: {sample_images_1f.shape}")
    print(f"Sample Batch Labels Shape: {sample_labels_1f.shape}")

    # 2. Sanity check for 3-frame dataset ("data_split_3frame") if present
    if resolve_data_root("data_split_3frame").exists():
        print("\n--- Testing 3-Frame Dataset ('data_split_3frame') ---")
        dataloaders_3frame = get_dataloaders("data_split_3frame", batch_size=32, num_workers=0)
        print(f"Train Dataset Size       : {len(dataloaders_3frame['train'].dataset)} images")
        print(f"Val Dataset Size         : {len(dataloaders_3frame['val'].dataset)} images")
        print(f"Test Dataset Size        : {len(dataloaders_3frame['test'].dataset)} images")

        sample_images_3f, sample_labels_3f = next(iter(dataloaders_3frame["train"]))
        print(f"Sample Batch Images Shape: {sample_images_3f.shape}")
        print(f"Sample Batch Labels Shape: {sample_labels_3f.shape}")

    print("\n" + "=" * 65)
    print("SANITY CHECK SUCCESSFUL: Stage 4 Dataset & DataLoaders Ready!")
    print("=" * 65)
