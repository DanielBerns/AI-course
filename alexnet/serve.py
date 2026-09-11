#!/usr/bin/env python3
import argparse
import os
import sys

import uvicorn
from src.utils.config import load_config, resolve_instance_paths
from src.web.app import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Modern AlexNet Web Studio & Inference Server")
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
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind server (0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on",
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
        config = load_config(config_file)
        config = resolve_instance_paths(config, instance_path)

        print(f"[*] Starting AlexNet Web Studio Server...")
        print(f"[*] Instance path   : {instance_path}")
        print(f"[*] Config          : {config_file}")
        print(f"[*] Network Endpoint: http://{args.host}:{args.port}")

        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except KeyboardInterrupt:
        print("\n[!] Server stopped by user. Shutting down.")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"[!] File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"[!] Missing required configuration key: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[!] Configuration error: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"[!] Network error (check host/port availability): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Unexpected error ({type(e).__name__}): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
