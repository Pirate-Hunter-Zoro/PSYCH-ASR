#!/bin/bash
#SBATCH --job-name=stage1b_sortformer_streaming
#SBATCH --partition=c3_accel
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -o slurm_jobs/logs/stage1b_sortformer_streaming_out.txt
#SBATCH -e slurm_jobs/logs/stage1b_sortformer_streaming_err.txt
#SBATCH --gres=gpu:1

# Stage 1b, ARM C: NVIDIA Sortformer, STREAMING (diar_streaming_sortformer_4spk-v2.1) -> RTTM.
#
# This is the only challenger whose weights are NOT non-commercial: NVIDIA Open Model
# License. Since the deliverable is an R21/R01, "which arm wins" and "which arm can ship"
# are two separate questions, and this arm is the one that can answer both.
#
# It also handles arbitrary recording length by construction, so unlike arm B it needs no
# windowing and pays no seam error. On a 50 minute session that may turn out to be its
# decisive advantage independent of accuracy.
#
# Runs in nemo_env, same as arm B -- both checkpoints load through SortformerEncLabelModel.
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
echo "Diarizing $(basename "${AUDIO_PATH}") with Sortformer (streaming)"
python scripts/diarize_sortformer.py "${AUDIO_PATH}" --mode streaming
