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
    pycocoevalcap \
    install-jdk

python - <<'PY'
import jdk
home = jdk.install('11')
open('/kaggle/working/.java_home', 'w').write(home)
print("JDK 11:", home)
PY

source "$(dirname "$0")/java_env.sh"
java -version 2>&1 | head -1

if ! command -v unzip >/dev/null 2>&1; then
    apt-get -qq update && apt-get -qq install -y unzip
fi

mkdir -p data/temp

python - <<'PY'
from pycocoevalcap.spice.spice import Spice
Spice().compute_score({"0": ["a man riding a horse"]}, {"0": ["a man riding a horse"]})
print("SPICE OK")
PY

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
import xmodaler.modeling, xmodaler.engine, xmodaler.evaluation
print("xmodaler imports OK")
PY
