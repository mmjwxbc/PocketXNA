#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# PocketXNA one-command training launcher.
#
# Usage:
#   docker run --rm --gpus all \
#       -v /path/to/data_train:/workspace/data_train \
#       pocketxmol:pxm_cu128 /opt/PocketXNA/docker/train_aptdesign.sh [args...]
#
# Or on the host (with pxm_cu128 already created):
#   ./docker/train_aptdesign.sh --num_gpus 2 --batch_size 32 [args...]
#
# Everything after the script name is passed through to scripts/train_pl.py,
# so the canonical PocketXMol flags all work. Convenience flags:
#   --config FILE        training config (default: configs/train/train_aptdesign.yml)
#   --num_gpus N         number of GPUs (default: 1, or all visible)
#   --batch_size N       override batch_size in the config
#   --max_steps N        override max_steps
#   --logdir DIR         log directory (default: lightning_logs)
#   --resume RUN         resume from lightning_logs/<RUN>/checkpoints/last.ckpt
#   --tag TAG            append a short tag to the run name
#   --clone              if PocketXNA isn't at /opt/PocketXNA (host), clone it
#                        from $POCKETXNA_REPO (default $PWD/../PocketXNA)
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POCKETXNA_REPO="${POCKETXNA_REPO:-https://github.com/mmjwxbc/PocketXNA.git}"
POCKETXNA_DIR="${POCKETXNA_DIR:-/opt/PocketXNA}"

# ----- parse launcher-only flags -------------------------------------------
CONFIG="configs/train/train_aptdesign.yml"
NUM_GPUS=""
BATCH_SIZE=""
MAX_STEPS=""
LOGDIR=""
RESUME=""
TAG=""
DO_CLONE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2;;
    --num_gpus)   NUM_GPUS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --max_steps)  MAX_STEPS="$2"; shift 2;;
    --logdir)     LOGDIR="$2"; shift 2;;
    --resume)     RESUME="$2"; shift 2;;
    --tag)        TAG="$2"; shift 2;;
    --clone)      DO_CLONE=1; shift;;
    --help|-h)
      sed -n '3,30p' "$0"; exit 0;;
    --) shift; break;;
    *)  break;;   # forward unknown flags to train_pl.py
  esac
done

# ----- detect / set up PocketXNA source -------------------------------------
if [[ ! -d "${POCKETXNA_DIR}" ]]; then
  if [[ "${DO_CLONE}" == 1 ]]; then
    echo "[train] cloning ${POCKETXNA_REPO} -> ${POCKETXNA_DIR}" >&2
    git clone --depth 1 "${POCKETXNA_REPO}" "${POCKETXNA_DIR}"
  elif [[ -d "${HOST_ROOT}/../PocketXNA" ]]; then
    POCKETXNA_DIR="${HOST_ROOT}/../PocketXNA"
    echo "[train] using local PocketXNA at ${POCKETXNA_DIR}" >&2
  else
    echo "[train] PocketXNA not found. Re-run with --clone to fetch it." >&2
    exit 1
  fi
fi

cd "${POCKETXNA_DIR}"

# ----- activate conda env (only when not already inside docker) ------------
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "pxm_cu128" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pxm_cu128
  else
    echo "[train] conda not on PATH and pxm_cu128 not active." >&2
    exit 1
  fi
fi

# ----- resolve GPU count -----------------------------------------------------
if [[ -z "${NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    NUM_GPUS=1
  fi
fi
echo "[train] num_gpus=${NUM_GPUS}" >&2

# ----- override config values via sed (lightweight, no PyYAML needed) -------
WORK_CFG="${CONFIG}"
TMP_CFG=""
if [[ -n "${BATCH_SIZE}" || -n "${MAX_STEPS}" ]]; then
  TMP_CFG="$(mktemp --suffix=.yml)"
  cp "${WORK_CFG}" "${TMP_CFG}"
  WORK_CFG="${TMP_CFG}"
  [[ -n "${BATCH_SIZE}" ]] && sed -i "s/^\(\s*batch_size:\s*\).*/\1${BATCH_SIZE}/" "${TMP_CFG}"
  [[ -n "${MAX_STEPS}"  ]] && sed -i "s/^\(\s*max_steps:\s*\).*/\1${MAX_STEPS}/"  "${TMP_CFG}"
  echo "[train] overrides applied to ${TMP_CFG}" >&2
fi
trap '[[ -n "${TMP_CFG}" ]] && rm -f "${TMP_CFG}"' EXIT

# ----- launch ---------------------------------------------------------------
RUN_NAME="aptdesign"
[[ -n "${TAG}" ]] && RUN_NAME="${RUN_NAME}_${TAG}"

LOGDIR_FINAL="${LOGDIR:-lightning_logs/${RUN_NAME}}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  exec torchrun \
       --standalone --nproc_per_node="${NUM_GPUS}" \
       scripts/train_pl.py \
         --config "${WORK_CFG}" \
         --num_gpus "${NUM_GPUS}" \
         --logdir  "${LOGDIR}" \
         --tag     "${TAG}" \
         "$@"
else
  exec python scripts/train_pl.py \
       --config "${WORK_CFG}" \
       --num_gpus 1 \
       --device 0 \
       --logdir  "${LOGDIR}" \
       --tag     "${TAG}" \
       "$@"
fi
