#!/bin/bash
# ---------------------------------------------------------------------------
# setup_envs.sh -- build all four conda prefix envs, each ending in its own
# import smoke-check. LOGIN NODE ONLY (it needs the internet).
#
#     bash scripts/setup_envs.sh
#
# WHY FOUR ENVS AND NOT ONE. whisperx 3.8.6's `~=` dependency chain LOCKS asr_env's
# torch, torchaudio, ctranslate2, faster-whisper and pyannote.audio versions. Any
# new dependency that pins torch itself is therefore not a package addition but an
# ENVIRONMENT FORK. Both challenger diarizers pin their own torch, so both are
# forks; the scorer is a fork for a different reason (see diar_eval_env below).
#
#   asr_env        3.11 / torch 2.8.0+cu128   whisperx, pyannote.audio, the join
#   diarizen_env   3.10 / torch 2.1.1+cu121   DiariZen + its vendored pyannote fork
#   nemo_env       3.11 / torch 2.13.0+cu130  nemo_toolkit[asr], both Sortformers
#   diar_eval_env  3.11 / NO TORCH            pyannote.metrics only, CPU
#
# NOTHING IS PIP INSTALLED FROM THIS REPO into any of them. Every entry point runs
# as `python -m psych_asr.cli.<name>` from the repo root, which is what puts the
# package on sys.path -- so a code change needs no reinstall in four envs, and the
# package's stdlib-only core stays importable from all four.
#
# Every step is idempotent: an env that already exists is reused, not rebuilt.
# ---------------------------------------------------------------------------

eval "$(/opt/apps/easybuild/software/Anaconda3/2025.06-0/bin/conda shell.bash hook)"

set -o errexit
set -o nounset
set -o pipefail

ENV_BASE="${PSYCH_ASR_VENV_ROOT:-/media/studies/ehr_study/analysis/mferguson/venvs}"
SRC_BASE="${PSYCH_ASR_SRC_ROOT:-/media/studies/ehr_study/analysis/mferguson/src}"
export CONDA_PKGS_DIRS="${ENV_BASE}/conda_pkgs"
mkdir -p "${SRC_BASE}"

# ---------------------------------------------------------------------------
# create_env <name> <python version>   -- make it if it is not already there
# enter_env  <name>                    -- activate, with PYTHONNOUSERSITE on
# leave_env                            -- deactivate
#
# PYTHONNOUSERSITE is not decoration. Without it, pip and python in a conda PREFIX
# env under /media/studies still see ~/.local/lib, where a stray user-site package
# shadows the pinned one and breaks the pin chain silently.
#
# The set +u / set -u wrap is not optional: conda's activation script reads unset
# variables and dies under nounset.
# ---------------------------------------------------------------------------
create_env() {
    local path="${ENV_BASE}/$1"
    if [ -f "${path}/conda-meta/history" ]; then
        echo "==== $1 already created"
    else
        echo "==== creating $1 (python $2)"
        conda create -p "${path}" "python=$2" -y
    fi
}

enter_env() {
    set +o nounset
    export PYTHONNOUSERSITE=1
    conda activate "${ENV_BASE}/$1"
    set -o nounset
}

leave_env() {
    set +o nounset; conda deactivate; set -o nounset
}

# ============================================================================
# asr_env -- ASR, alignment, the community-1 baseline diarizer, and the join.
# The pin set here is LOCKED by whisperx's ~= chain. Do not move it.
# The node's CUDA-13.3 driver runs the cu12 wheels (backward-compatible), and
# ctranslate2's cuDNN 9 rides in with torch, so no system cuDNN is required.
# ============================================================================
create_env asr_env 3.11
enter_env asr_env

python -m pip install \
    whisperx==3.8.6 \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchvision==0.23.0 \
    ctranslate2==4.8.1 \
    faster-whisper==1.2.1 \
    pyannote.audio==4.0.7 \
    transformers==4.55.4 \
    numpy==2.2.6 \
    pandas==2.3.1 \
    soundfile \
    python-dotenv==1.1.1 \
    pytest \
    ipykernel

echo "Verifying asr_env..."
python -c "
import ctranslate2, faster_whisper, numpy, pandas, pyannote.audio, soundfile, torch, torchaudio, transformers, whisperx
from importlib.metadata import version
print(f'asr_env OK | whisperx {version(\"whisperx\")} | torch {torch.__version__} | '
      f'cuda available: {torch.cuda.is_available()} | ctranslate2 {ctranslate2.__version__}')
"
leave_env

# ============================================================================
# diarizen_env -- arm A. torch 2.1.1 against asr_env's 2.8.0 is a hard conflict.
#
# DIARIZEN IS A SOURCE INSTALL, NOT A PACKAGE. It is not on PyPI, and it vendors
# its own pyannote-audio fork IN-TREE rather than depending on the released one --
# upstream's SpeakerDiarization will not accept the raw powerset segmentation
# checkpoint DiariZen's subclass needs. Hence the clone.
#
# The only true git submodule in that repo is dscore, and it is not needed:
# scoring happens in diar_eval_env with pyannote.metrics.
#
# 3.10 is what upstream builds against, and torch 2.1.1 has no 3.12 wheels.
# ============================================================================
DIARIZEN_SRC="${SRC_BASE}/DiariZen"
if [ -d "${DIARIZEN_SRC}/.git" ]; then
    echo "==== DiariZen source already cloned"
else
    git clone https://github.com/BUTSpeechFIT/DiariZen.git "${DIARIZEN_SRC}"
fi

create_env diarizen_env 3.10
enter_env diarizen_env

# cu121 wheels on the node's CUDA 13.3 driver -- the same backward-compatibility
# bet asr_env already makes with cu12 torch 2.8.
python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121

# EVERY STEP AFTER THE TORCH TRIO PASSES -c constraints.txt, which re-pins torch
# and numpy so that neither the requirements nor the vendored fork can drag torch
# forward underneath the install.
python -m pip install -r "${DIARIZEN_SRC}/requirements.txt" -c "${DIARIZEN_SRC}/constraints.txt"
python -m pip install -e "${DIARIZEN_SRC}"                  -c "${DIARIZEN_SRC}/constraints.txt"
# The VENDORED fork, not upstream pyannote.audio. Upstream's own instructions add
# [dev,testing]; those are lint/test extras only and are skipped here.
python -m pip install -e "${DIARIZEN_SRC}/pyannote-audio"   -c "${DIARIZEN_SRC}/constraints.txt"

echo "Verifying diarizen_env..."
python -c "
import numpy, pyannote.audio, toml, torch
from diarizen.pipelines.inference import DiariZenPipeline
from diarizen.models.module.wavlm_config import get_config
get_config('wavlm_large_s80_md')
print(f'diarizen_env OK | torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | '
      f'numpy {numpy.__version__} | pyannote.audio {pyannote.audio.__version__}')
"
leave_env

# ============================================================================
# nemo_env -- arms B and C. Both Sortformer checkpoints load through the same
# class, so one env covers two arms. NeMo brings hydra, lightning, omegaconf and
# its own transformers pin -- exactly the set that would break asr_env's locked
# transformers 4.55.4.
# ============================================================================
create_env nemo_env 3.11
enter_env nemo_env

# Cython and packaging must land BEFORE nemo_toolkit: several of its ASR
# dependencies have no wheels and build their setup.py against Cython at install
# time.
python -m pip install Cython packaging
python -m pip install "nemo_toolkit[asr]==2.7.3"

echo "Verifying nemo_env..."
python -c "
import nemo, torch
from nemo.collections.asr.models import SortformerEncLabelModel
from importlib.metadata import version
print(f'nemo_env OK | nemo {version(\"nemo_toolkit\")} | torch {torch.__version__} | '
      f'cuda available: {torch.cuda.is_available()}')
"
leave_env

# ============================================================================
# diar_eval_env -- the scorer, DELIBERATELY TORCH-FREE. CPU only; no GPU job ever
# activates it.
#
# This is the one that looks like over-engineering and is not. Folding the scorer
# into asr_env risks bumping pyannote.core underneath the locked pin set, and it
# quietly couples "how we measure" to "what we measure with." The scorer must be
# ONE implementation at ONE collar across every arm, or the comparison measures
# the scorer instead of the models.
#
# typing_extensions is imported by pyannote.metrics and NOT declared as one of its
# dependencies, so a clean install of pyannote.metrics alone fails its own first
# import. Do not remove it on the grounds that nothing appears to use it.
# ============================================================================
create_env diar_eval_env 3.11
enter_env diar_eval_env

python -m pip install pyannote.metrics==3.2.1 typing_extensions numpy==2.2.6 pandas==2.3.1 pytest

echo "Verifying diar_eval_env..."
python -c "
import importlib.util, numpy, pandas, pyannote.metrics
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from importlib.metadata import version
print(f'diar_eval_env OK | pyannote.metrics {version(\"pyannote.metrics\")} | '
      f'torch absent: {importlib.util.find_spec(\"torch\") is None}')
"
leave_env

echo ""
echo "All four environments built and verified."
