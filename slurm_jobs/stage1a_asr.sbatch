#!/bin/bash
#SBATCH --job-name=stage1a_asr
#SBATCH --partition=c3_accel
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/logs/stage1a_asr_out.txt
#SBATCH -e slurm_jobs/logs/stage1a_asr_err.txt
#SBATCH --gres=gpu:1

# Stage 1a: ASR + forced alignment, ONCE per session. Every bake-off arm fans off the
# aligned transcript this job writes, so that a DER difference between arms is a
# difference between diarizers and not between transcripts.
#
# Submit FROM THE REPO ROOT -- every path below is relative to the submit directory.

set -e

CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/asr_env
INPUT_DIR=data/inbox

module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME="/media/studies/ehr_study/analysis/mferguson/models/torch_home"
export NLTK_DATA="/media/studies/ehr_study/analysis/mferguson/models/nltk_data"

echo "Pinning to GPU ${CUDA_VISIBLE_DEVICES} on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

WAV_COUNT=$(ls "${INPUT_DIR}"/*.wav | wc -l)
if [[ "${WAV_COUNT}" -ne 1 ]]; then
    echo "Make sure ${INPUT_DIR} has exactly 1 .wav file in it"
    exit 1
fi
AUDIO_PATH=$(ls "${INPUT_DIR}"/*.wav)
echo "Processing $(basename "${AUDIO_PATH}")"
python scripts/run_asr.py "${AUDIO_PATH}"
