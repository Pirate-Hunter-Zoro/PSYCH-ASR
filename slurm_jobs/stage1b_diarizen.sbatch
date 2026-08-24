#!/bin/bash
#SBATCH --job-name=stage1b_diarizen
#SBATCH --partition=c3_accel
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -o slurm_jobs/logs/stage1b_diarizen_out.txt
#SBATCH -e slurm_jobs/logs/stage1b_diarizen_err.txt
#SBATCH --gres=gpu:1

# Stage 1b, ARM A: DiariZen (WavLM + Conformer EEND, powerset over 4 speakers) -> RTTM.
# WEIGHTS ARE CC BY-NC 4.0. Fine for feasibility research; not shippable.
#
# Runs in diarizen_env, NOT asr_env: DiariZen pins torch 2.1.1 against asr_env's locked
# 2.8.0, which is a hard conflict and the reason this arm is an environment fork.
#
# Wall time and memory are both raised over the baseline job. DiariZen slides a 16 s
# window at a 0.1 step, so a 50 minute session becomes a few thousand overlapping chunks,
# each of which is embedded and clustered -- far more forward passes than community-1's
# pipeline makes, on host memory as well as GPU.
#
# Submit FROM THE REPO ROOT.

set -e

CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/diarizen_env
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
echo "Diarizing $(basename "${AUDIO_PATH}") with DiariZen"
python scripts/diarize_diarizen.py "${AUDIO_PATH}" --num-speakers 2
