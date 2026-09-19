# COS-Net

A trimmed-down fork of [X-modaler](https://github.com/YehLi/xmodaler) kept to exactly what is needed to reproduce **COS-Net** on MS-COCO (Karpathy split).

Everything unrelated to COS-Net image captioning (video captioning, VQA, retrieval, the other
meta-architectures and datasets) has been removed.

## What is where

| Path | What it is |
| --- | --- |
| [configs/image_caption/cosnet/README.md](configs/image_caption/cosnet/README.md) | **Upstream COS-Net doc** - paper, BibTeX, download links for the pretrained features/annotations/checkpoints. |
| [xmodaler/datasets/README.md](xmodaler/datasets/README.md) | **Upstream dataset doc** - expected on-disk layout of `open_source_dataset/mscoco_dataset` and which config keys point at it. |
| [kaggle/KAGGLE.md](kaggle/KAGGLE.md) | **This fork's Kaggle runbook** - the notebook cells actually used to train and evaluate here, including how to resume across sessions. |
| [configs/image_caption/cosnet/COS-Net-preprocess/](configs/image_caption/cosnet/COS-Net-preprocess/) | Preprocessing steps (CLIP MIL scores, sentence retrieval, attribute labels) that build the COS-Net annotation `.pkl` files. |
| [kaggle/](kaggle/) | `setup.sh`, `prepare_data.py` (link the Kaggle dataset into place), `run_train.sh` / `run_train_rl.sh`, and the two override configs. |

Start with the upstream docs for *what the model and data are*; use the Kaggle runbook for
*how this fork is actually run*.

## Running on Kaggle (2x T4)

See [kaggle/KAGGLE.md](kaggle/KAGGLE.md) for the notebook cells.
