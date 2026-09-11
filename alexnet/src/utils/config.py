import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates YAML configuration file."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure required sections exist
    required_sections = ["dataset", "model", "training", "optimizer", "scheduler"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required section '{section}' in config file: {config_path}")

    return config


def resolve_instance_paths(config: Dict[str, Any], instance_path: str) -> Dict[str, Any]:
    """
    Overrides config['paths'] with absolute paths derived from instance_path.

    Creates the following sub-directories inside instance_path if they don't exist:
      data/, logs/, plots/, weights/

    This function always wins — any 'paths' section already in the YAML is discarded.
    """
    instance_path = os.path.abspath(os.path.expanduser(instance_path))
    sub_dirs = {
        "data_dir": "data",
        "log_dir": "logs",
        "plots_dir": "plots",
        "weights_dir": "weights",
    }

    paths: Dict[str, str] = {}
    for key, name in sub_dirs.items():
        full_path = os.path.join(instance_path, name)
        os.makedirs(full_path, exist_ok=True)
        paths[key] = full_path

    config["paths"] = paths
    return config


def find_latest_checkpoint(weights_dir: str) -> str | None:
    """
    Scans weights_dir for PyTorch checkpoint files (.pt).

    Returns the absolute path of the most recently modified file that contains
    a 'model_state_dict' key, or None if no compatible checkpoint is found.
    """
    import torch

    weights_dir = os.path.abspath(os.path.expanduser(weights_dir))
    if not os.path.isdir(weights_dir):
        return None

    candidates = sorted(
        [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith(".pt")],
        key=os.path.getmtime,
        reverse=True,
    )

    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                return path
        except Exception:
            continue

    return None


def save_config(config: Dict[str, Any], save_path: str):
    """Saves dictionary configuration back to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
