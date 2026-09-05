#!/bin/bash
#SBATCH --job-name=stage1_whisperx
#SBATCH --partition=c3_accel
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/logs/stage1_whisperx_out.txt
#SBATCH -e slurm_jobs/logs/stage1_whisperx_err.txt
#SBATCH --gres=gpu:1

# SINGLE-JOB Stage 1: ASR -> align -> diarize -> join -> render, in one process against
# the incumbent diarizer. Use the 1a/1b/1c chain (run_bakeoff.sh) for anything comparing
# diarizers; this is the path for one session with no dependency graph to submit, and it
# is the shape that produced the regression fixture.
#
# Clone this file for a new single-step GPU job.
#
# Submit FROM THE REPO ROOT.

set -e
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env asr_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Processing $(basename "${AUDIO_PATH}")"
python -m psych_asr.cli.run_whisperx "${AUDIO_PATH}"
