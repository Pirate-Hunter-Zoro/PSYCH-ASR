#!/bin/bash
# ---------------------------------------------------------------------------
# stage_models.sh -- download every offline model asset, ONCE, on the LOGIN NODE.
#
#     bash scripts/stage_models.sh
#
# LOGIN NODE ONLY. The login node has outbound internet; the compute nodes do
# not. Everything here is downloaded into a models/ directory on study storage
# and thereafter loaded by ABSOLUTE PATH with HF_HUB_OFFLINE=1 set in every job.
#
# Three of these are NOT Hugging Face downloads and are not covered by
# HF_HUB_OFFLINE, which is exactly why they are easy to forget:
#   * the torchaudio wav2vec2 alignment bundle, fetched from download.pytorch.org
#     into TORCH_HOME by scripts' warm_align_cache entry point
#   * nltk's punkt_tab, which WhisperX's alignment step loads for sentence
#     splitting and would otherwise nltk.download() from a compute node
#   * DiariZen's WavLM, which is NOT a download at all -- see below
#
# VERIFY A STAGED MODEL, DO NOT ASSUME IT. `hf download` can terminate leaving a
# directory holding only README.md and empty weight subfolders, with lock files
# under .cache/huggingface/download/ as the only trace. It looks staged from a
# casual ls. Confirm by file count AND an actual offline load.
# ---------------------------------------------------------------------------

eval "$(/opt/apps/easybuild/software/Anaconda3/2025.06-0/bin/conda shell.bash hook)"

set -o errexit
set -o nounset
set -o pipefail

# Derived from this script's own location rather than hardcoded, so a clone
# anywhere stages into its own tree.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_BASE="${PSYCH_ASR_VENV_ROOT:-/media/studies/ehr_study/analysis/mferguson/venvs}"
MODELS_ROOT="${PSYCH_ASR_MODELS_ROOT:-/media/studies/ehr_study/analysis/mferguson/models}"
export CONDA_PKGS_DIRS="${ENV_BASE}/conda_pkgs"

set +o nounset
export PYTHONNOUSERSITE=1
conda activate "${ENV_BASE}/asr_env"
set -o nounset

# ---- the token, exported so the `hf` CLI actually inherits it ----
# WITHOUT allexport the token stays shell-local and the download 401s. The .env
# is gitignored; never commit it.
ENV_PATH="${REPO_ROOT}/.env"
if [[ ! -f "${ENV_PATH}" ]]; then
    echo "${ENV_PATH} missing -- create it with HF_TOKEN=<read-scoped token>" >&2
    exit 1
fi
set -o allexport
source "${ENV_PATH}"
set +o allexport

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN empty in ${ENV_PATH}" >&2
    exit 1
fi

# For the DOWNLOAD only. The later loads all set it back on.
unset HF_HUB_OFFLINE

# ---- Hugging Face repos, staged by --local-dir and loaded by absolute path ----
# GATED: only the pyannote diarizer. Accept its terms once while logged in as the
# token's account; community-1 is auto-gated so access is instant.
#
#   A  DiariZen: 278 MB pytorch_model.bin holds the pruned WavLM + Conformer EEND
#      model. Its config.toml names wavlm_src "wavlm_large_s80_md", which looks
#      like a repo id and is NOT -- it resolves to a hardcoded dict in
#      diarizen/models/module/wavlm_config.py, and the pruned weights are already
#      inside that .bin. Nothing to stage.
#      Its speaker embedder is pyannote/wespeaker-voxceleb-resnet34-LM, staged
#      below for the community-1 baseline too, and handed in as a FILE path.
#   B  Sortformer offline: ships BOTH a .nemo archive and transformers-native
#      safetensors. We load the .nemo through SortformerEncLabelModel.restore_from,
#      because arm C ships no safetensors and ONE loading path across both arms is
#      worth more than saving 500 MB.
#   C  Streaming Sortformer: .nemo only.
stage () {   # stage <repo id> <local directory name>
    echo "==== staging $1 -> ${MODELS_ROOT}/$2"
    hf download "$1" --local-dir "${MODELS_ROOT}/$2"
}

stage Systran/faster-whisper-large-v3                  faster-whisper-large-v3
stage pyannote/speaker-diarization-community-1         pyannote-speaker-diarization-community-1
stage pyannote/wespeaker-voxceleb-resnet34-LM          pyannote-wespeaker-voxceleb-resnet34-LM
stage BUT-FIT/diarizen-wavlm-large-s80-md-v2           diarizen-wavlm-large-s80-md-v2
stage nvidia/diar_sortformer_4spk-v1                   diar_sortformer_4spk-v1
stage nvidia/diar_streaming_sortformer_4spk-v2.1       diar_streaming_sortformer_4spk-v2.1

# ---- the two caches that are not Hugging Face ----
export TORCH_HOME="${MODELS_ROOT}/torch_home"
export NLTK_DATA="${MODELS_ROOT}/nltk_data"
mkdir -p "${TORCH_HOME}" "${NLTK_DATA}"

# Run from the repo root: nothing is pip installed, so the repo landing on
# sys.path is what makes `python -m psych_asr...` resolve.
cd "${REPO_ROOT}"
python -m psych_asr.cli.warm_align_cache

# NOTE: english.pickle DOES NOT EXIST as a file. punkt_tab ships per-language
# folders of .tab/.txt data and NLTK synthesizes a PunktTokenizer from them when
# that name is requested. The absence of a .pickle on disk is not a failed
# download.
python -m nltk.downloader -d "${NLTK_DATA}" punkt_tab

echo ""
echo "Staged into ${MODELS_ROOT}. Confirm by file count and an offline load before"
echo "treating any of these as available."
