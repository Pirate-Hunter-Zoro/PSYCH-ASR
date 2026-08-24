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

# ---- Diarization bake-off challengers ----
# Every arm is staged by --local-dir and loaded by absolute path, same discipline as
# the ASR weights above. None of the three is gated, so no click-through is needed.
#   A  DiariZen: 278 MB pytorch_model.bin holds the pruned WavLM + Conformer EEND model.
#      Its config.toml names wavlm_src "wavlm_large_s80_md", which resolves to a HARDCODED
#      dict in diarizen/models/module/wavlm_config.py -- it is NOT a second download.
#      Its speaker embedder is pyannote/wespeaker-voxceleb-resnet34-LM, already staged
#      below for the community-1 baseline, and is handed in as a file path.
#   B  Sortformer offline: ships BOTH a .nemo archive and transformers-native safetensors.
#      We load the .nemo through SortformerEncLabelModel.restore_from.
#   C  Streaming Sortformer: .nemo only.
hf download BUT-FIT/diarizen-wavlm-large-s80-md-v2 \
    --local-dir "${MODELS_ROOT}/diarizen-wavlm-large-s80-md-v2"
hf download pyannote/wespeaker-voxceleb-resnet34-LM \
    --local-dir "${MODELS_ROOT}/pyannote-wespeaker-voxceleb-resnet34-LM"
hf download nvidia/diar_sortformer_4spk-v1 \
    --local-dir "${MODELS_ROOT}/diar_sortformer_4spk-v1"
hf download nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --local-dir "${MODELS_ROOT}/diar_streaming_sortformer_4spk-v2.1"

TORCH_CACHE="${MODELS_ROOT}/torch_home"
export TORCH_HOME="${TORCH_CACHE}"
mkdir -p "${TORCH_CACHE}"
NLTK_DIR="${MODELS_ROOT}/nltk_data"
export NLTK_DATA="${NLTK_DIR}"
mkdir -p "${NLTK_DIR}"

python "${PSYCH_ASR_PATH}/scripts/warm_align_cache.py"
python -m nltk.downloader -d "${NLTK_DIR}" punkt_tab