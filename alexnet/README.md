# AlexNet Pipeline

A modernized AlexNet training, dataset curation, and local web inference pipeline.

## Features

- **ModernAlexNet** — Original AlexNet architecture upgraded with BatchNorm, AdaptiveAvgPool2d, configurable dropout, and Kaiming weight initialization.
- **Flexible Dataset Sources** — Automatically downloads CIFAR-10/CIFAR-100, accepts custom local folders, or fetches a ZIP archive from a URL.
- **Instance-based Workflow** — All artefacts for a run (config, data, weights, logs, plots) live inside a single self-contained directory.
- **Automatic Checkpoint Resume** — On startup `train.py` scans the `weights/` directory and resumes from the latest compatible checkpoint automatically; starts from scratch if none is found.
- **Diagnostic Monitor** — Real-time overfitting, underfitting, and generalization-gap alerts each epoch.
- **Web Studio** — FastAPI-powered local UI for dataset curation (upload + gallery) and model inference with optional Grad-CAM visualizations.

## Installation

```bash
uv sync
```

## Quick Start

### 1 — Create an instance directory

```bash
mkdir -p /path/to/my_run
cp configs/config.yaml /path/to/my_run/
```

Edit `/path/to/my_run/config.yaml` to set your dataset source, epochs, batch size, and any other hyperparameters.
The four runtime sub-directories (`data/`, `logs/`, `plots/`, `weights/`) are **created automatically** — you do not need to make them yourself.

### 2 — Train

```bash
uv run python train.py --instance_path /path/to/my_run
```

On the first run the model is initialised from scratch.
On every subsequent run `train.py` scans `weights/` for the most recently modified compatible `.pt` file and **resumes automatically**.

### 3 — Serve

```bash
uv run python serve.py --instance_path /path/to/my_run
# Optional: override host / port
uv run python serve.py --instance_path /path/to/my_run --host 0.0.0.0 --port 8080
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Instance Directory Layout

```
/path/to/my_run/
├── config.yaml      ← required — copy from configs/config.yaml and edit
├── data/            ← dataset storage         (auto-created)
├── logs/            ← training logs           (auto-created)
├── plots/           ← training curve images   (auto-created)
└── weights/         ← model checkpoints       (auto-created)
```

You can maintain **multiple independent runs** by using separate instance directories:

```bash
uv run python train.py --instance_path /experiments/cifar10_run1
uv run python train.py --instance_path /experiments/cifar100_run2
```

## Project Structure

```
alexnet/
├── configs/
│   ├── config.yaml          # Template — copy into your instance directory
│   └── test_config.yaml     # Lightweight smoke-test configuration
├── src/
│   ├── dataset/
│   │   ├── custom_dataset.py   # CustomFolderDataset + DataLoader factory
│   │   └── downloader.py       # CIFAR-10/100 + URL + local dataset prep
│   ├── models/
│   │   └── alexnet.py          # ModernAlexNet definition
│   ├── training/
│   │   ├── trainer.py          # Training engine
│   │   ├── logger.py           # Console + file logger
│   │   └── diagnostics.py      # Overfitting / underfitting monitor
│   ├── utils/
│   │   ├── config.py           # YAML loader, instance-path resolver, checkpoint finder
│   │   ├── gradcam.py          # Grad-CAM heatmap generator
│   │   └── visualization.py    # Training curve plotter
│   └── web/
│       ├── app.py              # FastAPI application factory
│       ├── static/             # Stylesheets and assets
│       └── templates/          # Jinja2 HTML templates
├── train.py       # Training entry point
├── serve.py       # Web server entry point
└── pyproject.toml
```

## Configuration Reference

See [`configs/config.yaml`](configs/config.yaml) for all available options with inline comments.
The `paths` section is **not required** in `config.yaml` — all paths are derived automatically from `--instance_path` and any value present in the file is ignored.
