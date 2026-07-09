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