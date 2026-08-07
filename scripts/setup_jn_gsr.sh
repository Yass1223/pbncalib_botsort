#!/usr/bin/env bash
#
# jn_pipeline_gsr provisioning for sn-gamestate-lightning (Kaggle / Lightning.ai).
#
# The JN package needs its OWN interpreter (python 3.10, torch 2.0.1+cu118,
# mmcv 2.0.1, numpy 1.25.2, PARSeq) -- incompatible with this repo's env by
# design. This script:
#
#   1. ensures the package is vendored at plugins/jn_gsr
#      (copies from $JN_SRC if given and the folder is missing)
#   2. builds the package's own venv via its setup_env.py  (~7 GB;
#      JN_VENV overridable -- setup_kaggle.py convention: put it on the
#      ephemeral scratch disk on Kaggle)
#   3. fetches the hash-checked checkpoints into plugins/jn_gsr/models:
#      DBNet++ (111 MB) + legibility ResNet-34 (85 MB) via fetch_weights.py,
#      PARSeq (382 MB, load-gated through strhub) via stage_weights.py
#   4. runs the environment check and the worker self-test
#
# Usage:
#   bash scripts/setup_jn_gsr.sh
#   JN_SRC=/path/to/jn_pipeline_gsr bash scripts/setup_jn_gsr.sh   # vendor first
#   JN_VENV=/kaggle/tmp/.venv_jn    bash scripts/setup_jn_gsr.sh   # Kaggle scratch
#
# Weights come from the Ynniss HuggingFace repos (sha256-verified by the package
# itself); internet must be ON, HF_TOKEN exported if the repos are private.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JN_PKG="${JN_PKG:-${REPO_ROOT}/plugins/jn_gsr}"
export JN_VENV="${JN_VENV:-${JN_PKG}/.venv_jn}"

echo "=================================================================="
echo " jn_pipeline_gsr setup"
echo "   package: ${JN_PKG}"
echo "   venv:    ${JN_VENV}"
echo "=================================================================="

# --- 1. vendored package present? ---------------------------------------------
if [ ! -f "${JN_PKG}/jn_recognizer.py" ]; then
  if [ -n "${JN_SRC:-}" ] && [ -f "${JN_SRC}/jn_recognizer.py" ]; then
    echo "==> Vendoring package from ${JN_SRC}"
    mkdir -p "${JN_PKG}"
    cp -r "${JN_SRC}/." "${JN_PKG}/"
  else
    echo "ERROR: package not found at ${JN_PKG}." >&2
    echo "  Copy jn_pipeline_gsr there (Windows:  robocopy /E ^" >&2
    echo "    C:\\...\\jn_pipeline_gsr  <repo>\\plugins\\jn_gsr )" >&2
    echo "  or re-run with JN_SRC=/path/to/jn_pipeline_gsr" >&2
    exit 1
  fi
fi
for f in predict_tracklets.py setup_env.py fetch_weights.py stage_weights.py; do
  [ -f "${JN_PKG}/${f}" ] || { echo "ERROR: ${JN_PKG}/${f} missing -- stale copy? Re-vendor." >&2; exit 1; }
done

cd "${JN_PKG}"

# --- 2. the package's own venv (its setup_env.py owns every pin) ---------------
PYBIN="${JN_VENV}/bin/python"
if [ ! -x "${PYBIN}" ]; then
  echo "==> Building the JN venv (this is the long step)"
  python3 setup_env.py
else
  echo "==> Venv present; verifying"
  python3 setup_env.py --check || { echo "==> Check failed; repairing"; python3 setup_env.py; }
fi

# --- 3. checkpoints (hash-checked by the package's own fetchers) ---------------
echo "==> Fetching DBNet++ + legibility checkpoints"
"${PYBIN}" fetch_weights.py --out-dir models
echo "==> Fetching + load-gating the PARSeq checkpoint"
"${PYBIN}" stage_weights.py --out models/koshkina_sn_parseq.ckpt

# --- 4. self-tests -------------------------------------------------------------
echo "==> Worker self-test (offline)"
"${PYBIN}" predict_tracklets.py --self-test
echo "==> Package offline self-tests"
"${PYBIN}" gsr_adapter.py
"${PYBIN}" jn_recognizer.py

cat <<EOF

==================================================================
 Done.
   worker      : ${JN_PKG}/predict_tracklets.py
   interpreter : ${PYBIN}
   models      : ${JN_PKG}/models

 The module config (configs/modules/jersey_number_detect/jn_gsr.yaml)
 expects the package at <repo>/plugins/jn_gsr. If JN_VENV was moved
 (e.g. Kaggle scratch), override at run time:
   tracklab -cn soccernet_jngsr \\
     modules.jersey_number_detect.cfg.venv_python=${PYBIN}

 GPU sharding is automatic (nvidia-smi): 2 workers on 2xT4,
 4 on 4xT4, 1 otherwise.
==================================================================
EOF
