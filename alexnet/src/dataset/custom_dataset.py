import os
import random
from typing import Tuple, List, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class CustomFolderDataset(Dataset):
    """
    Image Dataset class supporting class capping, resizing, and augmentations.
    Directory structure expected: dataset_dir / <class_name> / image_file
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._scan_directory(max_samples_per_class)

    def _scan_directory(self, max_samples_per_class: Optional[int]):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")

        subdirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        if not subdirs:
            raise ValueError(f"No class subdirectories found in dataset path: {self.root_dir}")

        self.classes = subdirs
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            
            image_files = [
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS
            ]
            
            if max_samples_per_class is not None and len(image_files) > max_samples_per_class:
                random.seed(42)
                image_files = random.sample(image_files, max_samples_per_class)

            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns train and val data transformations."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_dataloaders(
    dataset_path: str,
    image_size: int = 224,
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 2,
    max_samples_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates train and validation DataLoaders from image folder directory.
    """
    train_transform, val_transform = get_transforms(image_size)

    # Scan once with train_transform to collect the canonical sample list.
    full_dataset = CustomFolderDataset(
        root_dir=dataset_path,
        transform=train_transform,
        max_samples_per_class=max_samples_per_class,
    )

    num_samples = len(full_dataset)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Build a second dataset view from the *same* samples list but with val_transform,
    # so validation indices always resolve to the identical images, just without augmentation.
    val_dataset = CustomFolderDataset(
        root_dir=dataset_path,
        transform=val_transform,
        max_samples_per_class=max_samples_per_class,
    )
    # Ensure the val view uses the same ordered samples as the train view
    # (both are derived from the same _scan_directory call pattern, seeded identically).
    # Overwrite only the val Subset's backing dataset — train_subset retains the original.
    val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.classes
