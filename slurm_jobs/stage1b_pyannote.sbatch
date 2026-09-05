#!/bin/bash
#SBATCH --job-name=stage1b_pyannote
#SBATCH --partition=c3_accel
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/logs/stage1b_pyannote_out.txt
#SBATCH -e slurm_jobs/logs/stage1b_pyannote_err.txt
#SBATCH --gres=gpu:1

# Stage 1b, BASELINE arm: pyannote speaker-diarization-community-1 -> RTTM.
# This is the incumbent and the regression reference: composing 1a -> this -> 1c must
# reproduce job 2032471's numbers exactly before any challenger arm means anything.
#
# Submit FROM THE REPO ROOT.

set -e
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env asr_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Diarizing $(basename "${AUDIO_PATH}") with community-1"
python -m psych_asr.cli.diarize_pyannote "${AUDIO_PATH}" --num-speakers 2
