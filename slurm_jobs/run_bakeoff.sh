#!/bin/bash

# Submit the whole diarization bake-off as one dependency chain.
#
#     1a ASR + align  (once)
#          |
#          +--> 1b community-1  (asr_env)       --+
#          +--> 1b diarizen     (diarizen_env)  --+
#          +--> 1b sortformer   (nemo_env)      --+--> 1c join + render (CPU, all arms)
#          +--> 1b sortformer-streaming (nemo_env)-+
#
# The four 1b jobs depend on 1a with afterok -- there is nothing to diarize against if the
# transcript never landed. 1c depends on the four with AFTERANY, not afterok, so one arm
# crashing still produces transcripts for the others. A dead arm shows up as a missing
# file, which is a result.
#
# Run FROM THE REPO ROOT. Requires exactly one .wav in data/inbox.

set -o errexit
set -o nounset
set -o pipefail

if [[ ! -d slurm_jobs || ! -d scripts ]]; then
    echo "Run this from the repo root -- every job's paths are relative to the submit directory."
    exit 1
fi

WAV_COUNT=$(ls data/inbox/*.wav 2>/dev/null | wc -l)
if [[ "${WAV_COUNT}" -ne 1 ]]; then
    echo "data/inbox must hold exactly 1 .wav file (found ${WAV_COUNT})."
    exit 1
fi
echo "Input: $(ls data/inbox/*.wav)"

mkdir -p slurm_jobs/logs

ASR_JOB=$(sbatch --parsable slurm_jobs/stage1a_asr.sbatch)
echo "1a ASR + align                : ${ASR_JOB}"

DIARIZE_JOBS=()
for ARM_JOB in stage1b_pyannote stage1b_diarizen stage1b_diarizen_free stage1b_sortformer stage1b_sortformer_streaming; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:"${ASR_JOB}" "slurm_jobs/${ARM_JOB}.sbatch")
    DIARIZE_JOBS+=("${JOB_ID}")
    echo "1b ${ARM_JOB}$(printf '%*s' $((30 - ${#ARM_JOB})) ''): ${JOB_ID}"
done

# Colon-separated list is what --dependency wants.
DEPENDENCY=$(IFS=:; echo "${DIARIZE_JOBS[*]}")
JOIN_JOB=$(sbatch --parsable --dependency=afterany:"${DEPENDENCY}" slurm_jobs/stage1c_join.sbatch)
echo "1c join + render              : ${JOIN_JOB}"

echo ""
echo "Watch with: squeue -u \$USER"
echo "Logs land in slurm_jobs/logs/ ; per-arm transcripts in data/stage1/<stem>.<arm>.transcript.txt"
