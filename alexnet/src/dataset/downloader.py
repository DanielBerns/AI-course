import os
import zipfile
import requests
from typing import Tuple, List, Optional
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from .custom_dataset import get_dataloaders


def prepare_cifar10(data_dir: str) -> str:
    """Downloads CIFAR-10 and converts it into folder structure."""
    target_folder = os.path.join(data_dir, "cifar10")
    if os.path.exists(target_folder) and len(os.listdir(target_folder)) == 10:
        return target_folder

    print(f"[*] Downloading CIFAR-10 dataset into: {target_folder}")
    raw_dir = os.path.join(data_dir, "raw_cifar10")
    cifar = CIFAR10(root=raw_dir, train=True, download=True)
    classes = cifar.classes

    for cls in classes:
        os.makedirs(os.path.join(target_folder, cls), exist_ok=True)

    for idx, (img, target) in enumerate(cifar):
        cls_name = classes[target]
        img_path = os.path.join(target_folder, cls_name, f"img_{idx:05d}.png")
        if not os.path.exists(img_path):
            img.save(img_path)

    return target_folder


def prepare_cifar100(data_dir: str) -> str:
    """Downloads CIFAR-100 and converts it into folder structure."""
    target_folder = os.path.join(data_dir, "cifar100")
    if os.path.exists(target_folder) and len(os.listdir(target_folder)) == 100:
        return target_folder

    print(f"[*] Downloading CIFAR-100 dataset into: {target_folder}")
    raw_dir = os.path.join(data_dir, "raw_cifar100")
    cifar = CIFAR100(root=raw_dir, train=True, download=True)
    classes = cifar.classes

    for cls in classes:
        os.makedirs(os.path.join(target_folder, cls), exist_ok=True)

    for idx, (img, target) in enumerate(cifar):
        cls_name = classes[target]
        img_path = os.path.join(target_folder, cls_name, f"img_{idx:05d}.png")
        if not os.path.exists(img_path):
            img.save(img_path)

    return target_folder


def download_and_extract_zip(url: str, output_dir: str) -> str:
    """Downloads a ZIP file from URL and extracts it."""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "downloaded_dataset.zip")
    
    print(f"[*] Downloading dataset from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[*] Extracting dataset to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    return output_dir


def prepare_dataset(config: dict) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    High-level utility to prepare dataset according to YAML configuration.
    """
    dataset_cfg = config["dataset"]
    paths_cfg = config["paths"]

    source = dataset_cfg.get("source", "cifar10").lower()
    data_dir = os.path.abspath(os.path.expanduser(paths_cfg.get("data_dir", "./data")))
    image_size = dataset_cfg.get("image_size", 224)
    batch_size = dataset_cfg.get("batch_size", 64)
    val_split = dataset_cfg.get("val_split", 0.2)
    num_workers = dataset_cfg.get("num_workers", 2)
    max_samples = dataset_cfg.get("max_samples_per_class", None)

    if source == "cifar10":
        folder_path = prepare_cifar10(data_dir)
    elif source == "cifar100":
        folder_path = prepare_cifar100(data_dir)
    elif source == "url":
        url = dataset_cfg.get("url", "")
        if not url:
            raise ValueError("Config 'dataset.source' is set to 'url' but 'dataset.url' is empty.")
        folder_path = download_and_extract_zip(url, os.path.join(data_dir, "downloaded"))
    elif source == "local":
        folder_path = dataset_cfg.get("local_path", "./data/curated")
        folder_path = os.path.abspath(os.path.expanduser(folder_path))
    else:
        raise ValueError(f"Unsupported dataset source: '{source}'. Choose 'cifar10', 'cifar100', 'local', or 'url'.")

    return get_dataloaders(
        dataset_path=folder_path,
        image_size=image_size,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        max_samples_per_class=max_samples,
        seed=config.get("training", {}).get("seed", 42),
    )
