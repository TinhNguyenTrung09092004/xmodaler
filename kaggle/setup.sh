#!/usr/bin/env bash
# Install the python deps X-modaler/COS-Net needs on top of a stock Kaggle GPU image.
# Run once per session, from the repo root.
set -e

pip install -q --no-cache-dir \
    fvcore \
    iopath \
    omegaconf \
    cloudpickle \
    tabulate \
    termcolor \
    portalocker \
    jsonlines \
    json_lines \
    pycocotools \
    pycocoevalcap

# xmodaler/tokenization imports pytorch_transformers at module load time (even though
# COS-Net uses a plain vocabulary.txt, not a BERT tokenizer). --no-deps keeps pip from
# touching the preinstalled torch.
pip install -q --no-cache-dir --no-deps pytorch_transformers
python - <<'PY'
import importlib, sys
for m in ("boto3", "requests", "regex", "six", "sentencepiece", "sacremoses"):
    try:
        importlib.import_module(m)
    except ImportError:
        print("missing dep for pytorch_transformers:", m)
PY

# pycocoevalcap shells out to java for PTBTokenizer / METEOR / SPICE.
if ! command -v java >/dev/null 2>&1; then
    echo "[WARN] java not found - COCOEvaler will crash during evaluation."
    echo "       apt-get -qq update && apt-get -qq install -y default-jre"
else
    java -version 2>&1 | head -1
fi

mkdir -p data/temp   # kfg.TEMP_DIR, relative to the repo root

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
import xmodaler, xmodaler.modeling, xmodaler.engine, xmodaler.evaluation
print("xmodaler imports OK")
PY
