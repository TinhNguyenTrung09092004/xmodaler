#!/usr/bin/env bash
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

if ! command -v java >/dev/null 2>&1; then
    echo "[WARN] java not found - evaluation will crash."
    echo "       apt-get -qq update && apt-get -qq install -y default-jre"
else
    java -version 2>&1 | head -1
fi

if ! command -v unzip >/dev/null 2>&1; then
    apt-get -qq update && apt-get -qq install -y unzip
fi

mkdir -p data/temp

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
import xmodaler.modeling, xmodaler.engine, xmodaler.evaluation
print("xmodaler imports OK")
PY
