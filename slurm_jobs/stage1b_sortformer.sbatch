#!/bin/bash
#SBATCH --job-name=stage1b_sortformer
#SBATCH --partition=c3_accel
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -o slurm_jobs/logs/stage1b_sortformer_offline_out.txt
#SBATCH -e slurm_jobs/logs/stage1b_sortformer_offline_err.txt
#SBATCH --gres=gpu:1

# Stage 1b, ARM B: NVIDIA Sortformer, OFFLINE (diar_sortformer_4spk-v1) -> RTTM.
# WEIGHTS ARE CC BY-NC 4.0. Fine for feasibility research; not shippable.
#
# Runs in nemo_env: NeMo brings its own torch, hydra, lightning and transformers pins.
#
# THIS ARM CANNOT SWALLOW A 50-MINUTE FILE. NVIDIA's model card puts the ceiling near
# 12 minutes on a 48 GB card and the A40 has 46 GB, so the script windows the audio and
# stitches speaker labels across the seams itself. That stitcher is machinery no other arm
# needs, and some of this arm's error will be seam error rather than model error -- which
# belongs in the writeup, not in a footnote.
#
# Submit FROM THE REPO ROOT.

set -e

CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/nemo_env
INPUT_DIR=data/inbox

module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

set +u; conda activate "${CONDA_ENV}"; set -u || exit 10
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME="/media/studies/ehr_study/analysis/mferguson/models/torch_home"

echo "Pinning to GPU ${CUDA_VISIBLE_DEVICES} on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

WAV_COUNT=$(ls "${INPUT_DIR}"/*.wav | wc -l)
if [[ "${WAV_COUNT}" -ne 1 ]]; then
    echo "Make sure ${INPUT_DIR} has exactly 1 .wav file in it"
    exit 1
fi
AUDIO_PATH=$(ls "${INPUT_DIR}"/*.wav)
echo "Diarizing $(basename "${AUDIO_PATH}") with Sortformer (offline)"
python scripts/diarize_sortformer.py "${AUDIO_PATH}" --mode offline
