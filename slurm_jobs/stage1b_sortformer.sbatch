#!/bin/bash
#SBATCH --job-name=stage1b_sortformer
#SBATCH --partition=c3_accel
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -o slurm_jobs/logs/stage1b_sortformer_offline_out.txt
#SBATCH -e slurm_jobs/logs/stage1b_sortformer_offline_err.txt
#SBATCH --gres=gpu:1

# Stage 1b, ARM B: NVIDIA Sortformer, OFFLINE (diar_sortformer_4spk-v1) -> RTTM.
# WEIGHTS ARE CC BY-NC 4.0. Fine for feasibility research; not shippable.
#
# Runs in nemo_env: NeMo brings its own torch, hydra, lightning and transformers pins.
#
# THIS ARM CANNOT SWALLOW A 50-MINUTE FILE. NVIDIA's model card puts the ceiling near
# 12 minutes on a 48 GB card and the A40 has 46 GB, so the run windows the audio and
# stitches speaker labels across the seams itself. That stitcher is machinery no other arm
# needs, and some of this arm's error will be seam error rather than model error -- which
# belongs in the writeup, not in a footnote.
#
# Submit FROM THE REPO ROOT.

set -e
source slurm_jobs/lib/job_env.sh

INPUT_DIR=data/inbox

activate_env nemo_env
report_gpu

AUDIO_PATH="$(sole_wav "${INPUT_DIR}")"
echo "Diarizing $(basename "${AUDIO_PATH}") with Sortformer (offline)"
python -m psych_asr.cli.diarize_sortformer "${AUDIO_PATH}" --mode offline
