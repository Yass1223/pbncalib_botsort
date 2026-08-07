#!/usr/bin/env bash
#
# Provision PnLCalib for the Option A calibration stage: clone the repo and
# fetch the v1.0.0 SV_kp / SV_lines checkpoints, verifying both.
#
# PnLCalib is GPL-2.0 and is NEVER vendored into this repository -- it is cloned
# at runtime, which is exactly what optiona_sfr's pnl_adapter.add_repo_to_path
# and Paths.pnlcalib_repo already expect.
#
#   bash scripts/setup_pnlcalib.sh
#   DEST=/kaggle/working/models bash scripts/setup_pnlcalib.sh
#   CHECK_ONLY=1 bash scripts/setup_pnlcalib.sh     # audit, download nothing
#
# Paths produced here match the defaults in
# sn_gamestate/configs/modules/calibration/optiona.yaml:
#   ${model_dir}/pnlcalib/PnLCalib          <- the clone (contains inference.py)
#   ${model_dir}/pnlcalib/weights/SV_kp     <- keypoint checkpoint
#   ${model_dir}/pnlcalib/weights/SV_lines  <- line checkpoint
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEST="${DEST:-pretrained_models/pnlcalib}"
REPO_DIR="${DEST}/PnLCalib"
W_DIR="${DEST}/weights"
GIT_URL="https://github.com/mguti97/PnLCalib.git"
BASE="https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"

# Published sizes, read from the GitHub release API.
declare -A SIZE=( ["SV_kp"]=264964645 ["SV_lines"]=264857893 )
MIN_BYTES=200000000

G="\033[0;32m"; R="\033[0;31m"; Y="\033[0;33m"; N="\033[0m"
ok()   { printf "  ${G}OK${N}    %s\n" "$1"; }
bad()  { printf "  ${R}MISS${N}  %s\n" "$1"; }
warn() { printf "  ${Y}WARN${N}  %s\n" "$1"; }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

FAILED=0

# ---------------------------------------------------------------- repo
hdr "PnLCalib repository"
# NOTE: clone PnLCalib ITSELF. A local reference copy may sit nested inside a
# wrapper directory (e.g. PnLCalib-main/PnLCalib-main/) because of how a GitHub
# zip extracts; that nesting is an artefact of the download, not the layout the
# code expects. pnl_adapter and inference.py both assume inference.py is at the
# TOP level of the path they are given, so REPO_DIR must contain inference.py
# directly. The check below enforces exactly that.
if [ -f "${REPO_DIR}/inference.py" ]; then
  ok "${REPO_DIR} (inference.py present)"
elif [ "${CHECK_ONLY:-0}" = "1" ]; then
  bad "${REPO_DIR}/inference.py absent"; FAILED=1
else
  mkdir -p "${DEST}"
  rm -rf "${REPO_DIR}"
  echo "  cloning ${GIT_URL}"
  if git clone --depth 1 "${GIT_URL}" "${REPO_DIR}"; then
    if [ -f "${REPO_DIR}/inference.py" ]; then
      ok "cloned to ${REPO_DIR}"
    else
      # Defensive: if a future layout nests the sources one level down, promote
      # them rather than leaving a path the pipeline cannot import from.
      inner=$(find "${REPO_DIR}" -maxdepth 2 -name inference.py -printf '%h\n' 2>/dev/null | head -1)
      if [ -n "${inner}" ]; then
        warn "inference.py was nested at ${inner}; flattening"
        mv "${inner}"/* "${REPO_DIR}/" 2>/dev/null
        ok "flattened to ${REPO_DIR}"
      else
        bad "clone succeeded but no inference.py found"; FAILED=1
      fi
    fi
  else
    bad "git clone failed"; FAILED=1
  fi
fi

# sanity: the two modules the adapter imports must be reachable
for f in "config/hrnetv2_w48.yaml" "config/hrnetv2_w48_l.yaml" \
         "model/cls_hrnet.py" "model/cls_hrnet_l.py" "utils/utils_calib.py"; do
  if [ -f "${REPO_DIR}/${f}" ]; then ok "${f}"; else bad "${f} absent"; FAILED=1; fi
done

# ---------------------------------------------------------------- weights
hdr "checkpoints (v1.0.0 release assets)"
mkdir -p "${W_DIR}"
for name in SV_kp SV_lines; do
  path="${W_DIR}/${name}"
  want="${SIZE[$name]}"
  if [ -f "${path}" ]; then
    got=$(stat -c%s "${path}" 2>/dev/null || echo 0)
    if [ "${got}" -ge "${MIN_BYTES}" ]; then
      if [ "${got}" = "${want}" ]; then
        ok "${name} (${got} bytes, exact)"
      else
        warn "${name} is ${got} bytes, release lists ${want} -- usable but unexpected"
      fi
      continue
    fi
    warn "${name} is ${got} bytes (truncated or an HTML error page); refetching"
    rm -f "${path}"
  fi
  if [ "${CHECK_ONLY:-0}" = "1" ]; then bad "${name} absent"; FAILED=1; continue; fi

  echo "  downloading ${name} (~$((want/1024/1024)) MiB)"
  # Atomic: fetch to .part, validate, then rename, so an interrupted download
  # can never masquerade as a good checkpoint on the next run.
  if curl -fL --retry 3 --retry-delay 2 -o "${path}.part" "${BASE}/${name}"; then
    got=$(stat -c%s "${path}.part" 2>/dev/null || echo 0)
    head2=$(head -c 2 "${path}.part" | tr -d '\0')
    if [ "${got}" -lt "${MIN_BYTES}" ]; then
      bad "${name}: only ${got} bytes -- not a checkpoint"; rm -f "${path}.part"; FAILED=1
    elif [ "${head2}" = "<!" ] || [ "${head2}" = "<h" ]; then
      bad "${name}: HTML error page, not a checkpoint"; rm -f "${path}.part"; FAILED=1
    else
      mv "${path}.part" "${path}"; ok "${name} (${got} bytes)"
    fi
  else
    bad "${name}: download failed"; rm -f "${path}.part"; FAILED=1
  fi
done

hdr "result"
if [ "${FAILED}" = "0" ]; then
  printf "  ${G}PnLCalib is provisioned.${N}\n"
  echo "  Point the config at it (these are optiona.yaml's defaults):"
  echo "    pnlcalib_repo: ${REPO_DIR}"
  echo "    weights_kp:    ${W_DIR}/SV_kp"
  echo "    weights_line:  ${W_DIR}/SV_lines"
  echo
  echo "  Next: python scripts/verify_pnlcalib_env.py --repo ${REPO_DIR} \\"
  echo "          --weights-kp ${W_DIR}/SV_kp --weights-line ${W_DIR}/SV_lines \\"
  echo "          --frame <a real frame>.jpg"
  exit 0
fi
printf "  ${R}provisioning incomplete${N} -- see MISS lines above\n"
exit 1
