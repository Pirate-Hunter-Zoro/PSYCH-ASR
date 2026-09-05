#!/bin/bash
#SBATCH --job-name=stage1b_diarizen_free
#SBATCH --partition=c3_accel
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -o slurm_jobs/logs/stage1b_diarizen_free_out.txt
#SBATCH -e slurm_jobs/logs/stage1b_diarizen_free_err.txt
#SBATCH --gres=gpu:1

# Stage 1b, ARM A-free: the SAME DiariZen model with its speaker count NOT pinned --
# the checkpoint's shipped clustering configuration decides for itself. Writes
# <stem>.diarizen-free.rttm, so it never collides with the pinned arm.
#
# This exists to answer a question rather than to win: the pyannote baseline is told there
# are exactly two people, and whether a challenger can be told the same thing is a real
# cost of switching. Running both ways measures what that constraint is worth on this
# audio instead of arguing about it.
#
# WEIGHTS ARE CC BY-NC 4.0. Same env and same resource reasoning as the pinned arm --
# see stage1b_diarizen.sbatch.
#
# Submit FROM THE REPO ROOT.

set -e
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env diarizen_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Diarizing $(basename "${AUDIO_PATH}") with DiariZen (shipped config, not pinned)"
python -m psych_asr.cli.diarize_diarizen "${AUDIO_PATH}" --num-speakers 0
