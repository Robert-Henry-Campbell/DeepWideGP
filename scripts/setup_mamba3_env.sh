#!/bin/bash
# Build the second Python environment, widedeepgp-m3, for the Mamba-3 sequence block.
#
# Why a second environment: published DeepGP uses the bidirectional Mamba block of the
# Vision-Mamba fork, which installs as `mamba_ssm`; Mamba-3 ships in upstream `mamba_ssm`
# (installed from source). The two packages share a name and cannot be installed together.
# Everything else is pinned to the versions of the existing widedeepgp environment.
#
#   bash scripts/setup_mamba3_env.sh [env_dir]      # default /mnt/sdg/robert/envs/widedeepgp-m3
#
# Afterwards:  /mnt/sdg/robert/envs/widedeepgp-m3/bin/python main_multilabel.py --mixer mamba3 ...
#
# Upstream mamba_ssm is installed WITHOUT its declared dependencies. Its metadata pulls in
# tilelang, apache-tvm-ffi and quack-kernels, which serve the multi-input (MIMO) and
# step-function kernels that DeepGP does not use, and which (11 Nov 2026):
#   * upgrade torch to 2.14 and numpy to 2.x, breaking the pins above;
#   * make `import mamba_ssm` fail on Python 3.10 (tvm_ffi raises AttributeError, which
#     mamba3.py's `except ImportError` does not catch).
# Without them the optional imports fall away cleanly and the single-input (SISO) kernels,
# which need only Triton (it comes with PyTorch), work.
set -euo pipefail
ENV=${1:-/mnt/sdg/robert/envs/widedeepgp-m3}
PYTHON=${PYTHON:-/usr/bin/python3.10}
# the upstream commit the gates were run on (mamba_ssm 2.3.2.post1)
MAMBA_REF=${MAMBA_REF:-e9594ce1c732d97440f0332fdc43170a2294dbfa}

[ -x "$ENV/bin/python" ] || "$PYTHON" -m venv "$ENV"
PIP="$ENV/bin/python -m pip"
# setuptools < 70: Lightning 2.0.8 imports pkg_resources, which later setuptools no longer ships
$PIP install --quiet --upgrade pip wheel "setuptools<70" packaging ninja
$PIP install --quiet torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install --quiet numpy==1.26.4 scipy==1.15.3 pandas==2.3.3 scikit-learn==1.7.2 einops==0.8.2 timm==1.0.24 \
    pytorch-lightning==2.0.8 torchmetrics==1.8.2 torcheval==0.0.7 tensorboard==2.20.0 optuna==5.0.0 \
    matplotlib==3.10.7 pytest==9.0.2 transformers==4.39.3
# Mamba-3 lives in the source tree, not in the PyPI wheel. --no-deps: see the header.
$PIP install --quiet --no-deps --no-build-isolation "git+https://github.com/state-spaces/mamba.git@${MAMBA_REF}"
$PIP install --quiet "setuptools<70"            # in case anything above moved it
"$ENV/bin/python" - <<'EOF'
import importlib.metadata as md
import importlib.util as iu

import torch

for unwanted in ("tilelang", "tvm_ffi"):
    assert iu.find_spec(unwanted) is None, f"{unwanted} is installed: `import mamba_ssm` fails with it on Python 3.10"
import mamba_ssm                    # noqa: F401
import pytorch_lightning            # noqa: F401  (fails without pkg_resources)
from mamba_ssm import Mamba3        # noqa: F401
print("torch", torch.__version__, "| numpy", md.version("numpy"), "| mamba_ssm", md.version("mamba-ssm"),
      "| triton", md.version("triton"), "| lightning", md.version("pytorch-lightning"))
assert torch.__version__.startswith("2.10.0") and md.version("numpy") == "1.26.4", "a dependency moved the pins"
print("Mamba3 imports")
EOF
