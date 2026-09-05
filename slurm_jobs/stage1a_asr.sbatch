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
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env asr_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Processing $(basename "${AUDIO_PATH}")"
python -m psych_asr.cli.run_asr "${AUDIO_PATH}"
