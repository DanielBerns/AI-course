#!/usr/bin/env python3
import argparse
import os
import sys

from src.utils.config import load_config, resolve_instance_paths, find_latest_checkpoint
from src.dataset.downloader import prepare_dataset
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Modern AlexNet End-to-End Training Pipeline")
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help=(
            "Absolute path to the instance directory. "
            "Must contain a 'config.yaml' file. "
            "Sub-directories data/, logs/, plots/, and weights/ are created automatically."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instance_path = os.path.abspath(os.path.expanduser(args.instance_path))

    if not os.path.isdir(instance_path):
        print(f"[!] Error: instance_path does not exist or is not a directory: {instance_path}")
        sys.exit(1)

    config_file = os.path.join(instance_path, "config.yaml")
    if not os.path.isfile(config_file):
        print(f"[!] Error: 'config.yaml' not found in instance directory: {instance_path}")
        sys.exit(1)

    try:
        print(f"[*] Instance path   : {instance_path}")
        print(f"[*] Loading config  : {config_file}")
        config = load_config(config_file)

        # Inject instance-derived absolute paths (always overrides any paths in the YAML)
        config = resolve_instance_paths(config, instance_path)

        weights_dir = config["paths"]["weights_dir"]

        # Auto-detect checkpoint for resume
        checkpoint_path = find_latest_checkpoint(weights_dir)
        if checkpoint_path:
            print(f"[*] Found compatible checkpoint — resuming: {os.path.basename(checkpoint_path)}")
            config["training"]["resume_checkpoint"] = checkpoint_path
        else:
            print("[*] No compatible checkpoint found — starting training from scratch.")
            config["training"]["resume_checkpoint"] = None

        # Prepare dataset
        print(f"[*] Preparing dataset (Source: {config['dataset']['source']}) ...")
        train_loader, val_loader, class_names = prepare_dataset(config)

        # Initialize Trainer and start training
        trainer = Trainer(config, train_loader, val_loader, class_names)
        trainer.train()

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Shutting down.")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"[!] File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"[!] Missing required configuration key: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[!] Configuration or data error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[!] Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Unexpected error ({type(e).__name__}): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
