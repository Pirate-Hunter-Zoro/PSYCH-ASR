#!/bin/bash

eval "$(/opt/apps/easybuild/software/Anaconda3/2025.06-0/bin/conda shell.bash hook)"

set -o errexit
set -o nounset
set -o pipefail

ENV_BASE="/media/studies/ehr_study/analysis/mferguson/venvs"
export CONDA_PKGS_DIRS=$ENV_BASE/"conda_pkgs"

# Create environment for main pipeline
ASR_ENV=$ENV_BASE/asr_env

if [ -f "$ASR_ENV/conda-meta/history" ]; then
    echo "Main pipeline environment already created"
else
    # prefix flag and auto-yes on proceeding
    conda create -p $ASR_ENV python=3.11 -y
fi

set +o nounset
export PYTHONNOUSERSITE=1
conda activate $ASR_ENV
set -o nounset

python -m pip install whisperx==3.8.6\
                    torch==2.8.0\
                    torchaudio==2.8.0\
                    torchvision==0.23.0\
                    ctranslate2==4.8.1\
                    faster-whisper==1.2.1\
                    pyannote.audio==4.0.7\
                    transformers==4.55.4\
                    numpy==2.2.6\
                    pandas==2.3.1\
                    soundfile\
                    python-dotenv==1.1.1\
                    pytest\
                    ipykernel

echo "Verifying main pipeline environment..."
python -c "import faster_whisper, pyannote.audio,  whisperx, torch, torchaudio, ctranslate2, soundfile, numpy, pandas, transformers; from importlib.metadata import version; print(f'main pipeline env OK | whisperx version: {version(\"whisperx\")} | torch version: {torch.__version__} | cuda available: {torch.cuda.is_available()} | ctranslate2 version: {ctranslate2.__version__}')"

set +o nounset; conda deactivate; set -o nounset

# ================================================================
# Diarization bake-off environments
# ================================================================
# Each challenger diarizer pins its OWN torch, so each is an environment FORK, not a
# package addition -- asr_env's pin set is locked by whisperx's `~=` chain and must not
# move. Every arm exchanges RTTM with the shared join step, so nothing here needs to
# import whisperx.

SRC_BASE="/media/studies/ehr_study/analysis/mferguson/src"
mkdir -p "${SRC_BASE}"

# ---------------- Arm A: DiariZen (torch 2.1.1) ----------------
# DiariZen is a SOURCE install: not on PyPI, and it vendors its own pyannote-audio fork
# in-tree (the only real git submodule is dscore, which we do not use -- scoring happens
# in diar_eval_env with pyannote.metrics). Hence the clone.
DIARIZEN_ENV="${ENV_BASE}/diarizen_env"
DIARIZEN_SRC="${SRC_BASE}/DiariZen"

if [ -d "${DIARIZEN_SRC}/.git" ]; then
    echo "DiariZen source already cloned"
else
    git clone https://github.com/BUTSpeechFIT/DiariZen.git "${DIARIZEN_SRC}"
fi

if [ -f "$DIARIZEN_ENV/conda-meta/history" ]; then
    echo "DiariZen environment already created"
else
    # 3.10 is what upstream builds against, and torch 2.1.1 has no 3.12 wheels.
    conda create -p $DIARIZEN_ENV python=3.10 -y
fi

set +o nounset
export PYTHONNOUSERSITE=1
conda activate $DIARIZEN_ENV
set -o nounset

# cu121 wheels on the node's CUDA 13.3 driver, same backward-compatibility bet asr_env
# already makes with cu12 torch 2.8.
python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121
# constraints.txt re-pins torch/numpy so neither the requirements nor the pyannote fork
# can drag torch forward underneath the install.
python -m pip install -r "${DIARIZEN_SRC}/requirements.txt" -c "${DIARIZEN_SRC}/constraints.txt"
python -m pip install -e "${DIARIZEN_SRC}" -c "${DIARIZEN_SRC}/constraints.txt"
# The vendored fork, NOT upstream pyannote.audio. Upstream's SpeakerDiarization does not
# accept a raw powerset segmentation checkpoint the way DiariZen's subclass needs.
# Upstream's own instructions add [dev,testing]; those are lint/test extras only and are
# skipped here.
python -m pip install -e "${DIARIZEN_SRC}/pyannote-audio" -c "${DIARIZEN_SRC}/constraints.txt"

echo "Verifying DiariZen environment..."
python -c "import torch, numpy, pyannote.audio, toml; from diarizen.pipelines.inference import DiariZenPipeline; from diarizen.models.module.wavlm_config import get_config; get_config('wavlm_large_s80_md'); print(f'diarizen env OK | torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | numpy {numpy.__version__} | pyannote.audio {pyannote.audio.__version__}')"

set +o nounset; conda deactivate; set -o nounset

# ---------------- Arms B and C: NeMo Sortformer ----------------
# Both Sortformer checkpoints load through the same class, so one env covers two arms.
# NeMo brings hydra, lightning, omegaconf and its own transformers pin -- exactly the
# set that would break asr_env's locked 4.55.4.
NEMO_ENV="${ENV_BASE}/nemo_env"

if [ -f "$NEMO_ENV/conda-meta/history" ]; then
    echo "NeMo environment already created"
else
    conda create -p $NEMO_ENV python=3.11 -y
fi

set +o nounset
conda activate $NEMO_ENV
set -o nounset

# Cython and packaging must land BEFORE nemo_toolkit: several of its ASR dependencies
# have no wheels and build their setup.py against Cython at install time.
python -m pip install Cython packaging
python -m pip install "nemo_toolkit[asr]==2.7.3"

echo "Verifying NeMo environment..."
python -c "import torch, nemo; from nemo.collections.asr.models import SortformerEncLabelModel; from importlib.metadata import version; print(f'nemo env OK | nemo {version(\"nemo_toolkit\")} | torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')"

set +o nounset; conda deactivate; set -o nounset

# ---------------- Scoring: pyannote.metrics, deliberately torch-free ----------------
# One scorer, one collar, every arm. Folding this into asr_env risks bumping
# pyannote.core underneath the locked pin set, and it couples "how we measure" to
# "what we measure with." CPU only -- no GPU job ever activates this env.
DIAR_EVAL_ENV="${ENV_BASE}/diar_eval_env"

if [ -f "$DIAR_EVAL_ENV/conda-meta/history" ]; then
    echo "Diarization scoring environment already created"
else
    conda create -p $DIAR_EVAL_ENV python=3.11 -y
fi

set +o nounset
conda activate $DIAR_EVAL_ENV
set -o nounset

# typing_extensions is imported by pyannote.metrics but NOT declared as one of its
# dependencies, so a clean install of just pyannote.metrics fails its own import.
python -m pip install pyannote.metrics==3.2.1 typing_extensions numpy==2.2.6 pandas==2.3.1 pytest

echo "Verifying diarization scoring environment..."
python -c "import pyannote.metrics, pandas, numpy; from pyannote.metrics.diarization import DiarizationErrorRate; from pyannote.database.util import load_rttm; from importlib.metadata import version; print(f'diar_eval env OK | pyannote.metrics {version(\"pyannote.metrics\")} | torch absent: {__import__(\"importlib.util\", fromlist=[\"util\"]).find_spec(\"torch\") is None}')"

set +o nounset; conda deactivate; set -o nounset

echo "All environments built and verified."