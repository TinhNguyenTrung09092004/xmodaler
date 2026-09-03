#!/usr/bin/env bash
# Install the deps X-modaler/COS-Net needs on top of a stock Kaggle GPU image.
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

# pytorch_transformers is deliberately not installed: xmodaler/tokenization falls back
# to a stub when it is absent, and COS-Net uses a plain vocabulary.txt, not a tokenizer.

# pycocoevalcap shells out to java for PTBTokenizer / METEOR / SPICE.
if ! command -v java >/dev/null 2>&1; then
    echo "[WARN] java not found - evaluation will crash."
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
import xmodaler.modeling, xmodaler.engine, xmodaler.evaluation
print("xmodaler imports OK")
PY
