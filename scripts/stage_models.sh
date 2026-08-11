#!/bin/bash

eval "$(/opt/apps/easybuild/software/Anaconda3/2025.06-0/bin/conda shell.bash hook)"

set -o errexit
set -o nounset
set -o pipefail

ENV_BASE="/media/studies/ehr_study/analysis/mferguson/venvs"
export CONDA_PKGS_DIRS=$ENV_BASE/"conda_pkgs"

# Create environment for main pipeline
ASR_ENV="${ENV_BASE}/asr_env"

set +o nounset
export PYTHONNOUSERSITE=1
conda activate $ASR_ENV
set -o nounset

MODELS_ROOT="/media/studies/ehr_study/analysis/mferguson/models"
MODEL_DIR="${MODELS_ROOT}/faster-whisper-large-v3"
REPO_ID="Systran/faster-whisper-large-v3"
PSYCH_ASR_PATH="/home/librad.laureateinstitute.org/mferguson/PSYCH-ASR"
ENV_PATH="${PSYCH_ASR_PATH}/.env"

if [[ ! -f "${ENV_PATH}" ]]; then
    echo ".env missing..."
    exit 1
fi

# All later executed programs will see all environment variables
set -o allexport
source "${ENV_PATH}"
set +o allexport

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN empty"
    exit 1
fi

unset HF_HUB_OFFLINE
hf download "${REPO_ID}" --local-dir "${MODEL_DIR}"

TORCH_CACHE="${MODELS_ROOT}/torch_home"
export TORCH_HOME="${TORCH_CACHE}"
mkdir -p "${TORCH_CACHE}"
NLTK_DIR="${MODELS_ROOT}/nltk_data"
export NLTK_DATA="${NLTK_DIR}"
mkdir -p "${NLTK_DIR}"

python "${PSYCH_ASR_PATH}/scripts/warm_align_cache.py"
python -m nltk.downloader -d "${NLTK_DIR}" punkt_tab