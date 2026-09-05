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

# Does the GPU / torch / ctranslate2 stack work at all? Transcribes a synthesized sine
# tone with faster-whisper-TINY, so it needs no session audio and no real weights.
#
# Submit FROM THE REPO ROOT.

set -e
source slurm_jobs/lib/job_env.sh

activate_env asr_env
report_gpu

TONE_WAV="${TMPDIR:-/tmp}/gpu_smoke.wav"
ffmpeg -y -f lavfi -i sine=frequency=440:duration=3 -ar 16000 -ac 1 -loglevel error "${TONE_WAV}"
python -m psych_asr.cli.gpu_smoke "${TONE_WAV}"
rm -f "${TONE_WAV}"
