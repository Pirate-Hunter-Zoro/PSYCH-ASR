#!/bin/bash
#SBATCH --job-name=gpu_smoke
#SBATCH --partition=c3_accel
#SBATCH --time=0-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/logs/gpu_smoke_out.txt
#SBATCH -e slurm_jobs/logs/gpu_smoke_err.txt
#SBATCH --gres=gpu:1

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/asr_env

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10
export PYTHONNOUSERSITE=1

# Pin to the GPU with the most free memory (works around bare-metal squatters slurm cannot see)
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -n -r | head -1 | awk -F, '{print $1}' | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
echo "Pinning to GPU ${CUDA_VISIBLE_DEVICES} on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

TONE_WAV="/tmp/gpu_smoke.wav"
ffmpeg -y -f lavfi -i sine=frequency=440:duration=3 -ar 16000 -ac 1 -loglevel error $TONE_WAV

python scripts/gpu_smoke.py $TONE_WAV