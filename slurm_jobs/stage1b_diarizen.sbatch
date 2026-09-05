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

# Stage 1b, ARM A: DiariZen (WavLM + Conformer EEND, powerset over 4 speakers) -> RTTM,
# with the speaker count PINNED to 2. The unpinned twin is stage1b_diarizen_free.sbatch.
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
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env diarizen_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Diarizing $(basename "${AUDIO_PATH}") with DiariZen (pinned to 2 speakers)"
python -m psych_asr.cli.diarize_diarizen "${AUDIO_PATH}" --num-speakers 2
