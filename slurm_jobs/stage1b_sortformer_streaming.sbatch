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
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env nemo_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Diarizing $(basename "${AUDIO_PATH}") with Sortformer (streaming)"
python -m psych_asr.cli.diarize_sortformer "${AUDIO_PATH}" --mode streaming
