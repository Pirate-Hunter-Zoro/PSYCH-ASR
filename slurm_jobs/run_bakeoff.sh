#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the whole diarization bake-off as one dependency chain.
#
#     1a ASR + align  (once)
#          |
#          +--> 1b community-1          (asr_env)      --+
#          +--> 1b diarizen             (diarizen_env) --+
#          +--> 1b diarizen-free        (diarizen_env) --+--> 1c join + render (CPU, all arms)
#          +--> 1b sortformer           (nemo_env)     --+
#          +--> 1b sortformer-streaming (nemo_env)     --+
#
# THE DEPENDENCY KINDS ARE NOT INTERCHANGEABLE. The 1b jobs depend on 1a with afterok --
# there is nothing to diarize against if the transcript never landed. 1c depends on the
# arms with AFTERANY, so one arm crashing still produces transcripts for the others; a
# dead arm then shows up as a missing file, which is a result rather than a silent gap.
#
# Run FROM THE REPO ROOT. Requires exactly one .wav in data/inbox.
# ---------------------------------------------------------------------------

set -o errexit
set -o nounset
set -o pipefail

# The arms, in submission order. ADDING A FIFTH ARM IS A LINE HERE plus its .sbatch --
# nothing downstream needs changing, because 1c walks whatever RTTMs exist and derives each
# arm name from the RTTM's own filename.
ARM_JOBS=(
    stage1b_pyannote
    stage1b_diarizen
    stage1b_diarizen_free
    stage1b_sortformer
    stage1b_sortformer_streaming
)

if [[ ! -d slurm_jobs || ! -d psych_asr ]]; then
    echo "Run this from the repo root -- every job's paths are relative to the submit directory." >&2
    exit 1
fi

WAVS=( data/inbox/*.wav )
if [[ ! -e "${WAVS[0]}" || "${#WAVS[@]}" -ne 1 ]]; then
    echo "data/inbox must hold exactly 1 .wav file (found $([[ -e "${WAVS[0]}" ]] && echo "${#WAVS[@]}" || echo 0))." >&2
    exit 1
fi
echo "Input: ${WAVS[0]}"

mkdir -p slurm_jobs/logs

ASR_JOB=$(sbatch --parsable slurm_jobs/stage1a_asr.sbatch)
printf '%-30s: %s\n' "1a ASR + align" "${ASR_JOB}"

DIARIZE_JOBS=()
for ARM_JOB in "${ARM_JOBS[@]}"; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:"${ASR_JOB}" "slurm_jobs/${ARM_JOB}.sbatch")
    DIARIZE_JOBS+=("${JOB_ID}")
    printf '%-30s: %s\n' "1b ${ARM_JOB}" "${JOB_ID}"
done

# Colon-separated list is what --dependency wants.
DEPENDENCY=$(IFS=:; echo "${DIARIZE_JOBS[*]}")
JOIN_JOB=$(sbatch --parsable --dependency=afterany:"${DEPENDENCY}" slurm_jobs/stage1c_join.sbatch)
printf '%-30s: %s\n' "1c join + render" "${JOIN_JOB}"

echo ""
echo "Watch with: squeue -u \$USER"
echo "Logs land in slurm_jobs/logs/ ; per-arm transcripts in data/stage1/<stem>.<arm>.transcript.txt"
