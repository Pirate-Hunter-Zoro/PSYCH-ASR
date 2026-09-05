#!/bin/bash
#SBATCH --job-name=stage1c_join
#SBATCH --partition=c3_short
#SBATCH --time=0-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o slurm_jobs/logs/stage1c_join_out.txt
#SBATCH -e slurm_jobs/logs/stage1c_join_err.txt

# Stage 1c: join every arm's RTTM onto the shared aligned transcript, render each, then
# run the regression gate and the cross-arm diff.
#
# NO GPU. This step loads no models -- it is a timestamp join and a text render, seconds of
# CPU per arm -- so it asks for no --gres and runs on c3_short rather than occupying the
# single c3_accel node the arms themselves need.
#
# It walks whatever RTTMs exist rather than a fixed arm list, and it does not stop when one
# is missing. That is deliberate: the driver makes this job depend on the 1b jobs with
# afterany, so one arm crashing still lets every other arm produce its transcript. A missing
# arm shows up as a missing file, which is a visible result rather than a silent one.
#
# Submit FROM THE REPO ROOT.

set -e
source slurm_jobs/lib/job_env.sh

STAGE1_DIR=data/stage1

activate_env asr_env

ALIGNED_PATH="$(sole_aligned "${STAGE1_DIR}")"
echo "Aligned transcript: ${ALIGNED_PATH}"

JOINED=0
FAILED=0
for TURN_TABLE in "${STAGE1_DIR}"/*.rttm; do
    [[ -e "${TURN_TABLE}" ]] || continue
    # The .exclusive.rttm is community-1's overlap-free view, kept for Stage 3a's overlap
    # features. It is a diagnostic on an existing arm, not an arm of its own.
    case "${TURN_TABLE}" in
        *.exclusive.rttm) echo "Skipping ${TURN_TABLE} (overlap-free view, not an arm)"; continue ;;
    esac

    echo ""
    echo "########## joining ${TURN_TABLE} ##########"
    if python -m psych_asr.cli.join_speakers "${ALIGNED_PATH}" "${TURN_TABLE}"; then
        JOINED=$((JOINED + 1))
    else
        echo "JOIN FAILED for ${TURN_TABLE}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Joined ${JOINED} arm(s); ${FAILED} failed."

# The regression gate: the baseline arm composed through the split must reproduce the
# single-job fixture from job 2032471. The fixture is the one .diarized.json with NO arm in
# its name, which is what the single-job path writes and the split never does. Reported,
# never silently skipped -- if the fixture is absent that is itself worth saying out loud.
FIXTURE=$(ls "${STAGE1_DIR}"/*.diarized.json 2>/dev/null | grep -v '\.[a-z0-9-]*\.diarized\.json$' | head -1 || true)
CANDIDATE=$(ls "${STAGE1_DIR}"/*.community-1.diarized.json 2>/dev/null | head -1 || true)
if [[ -n "${FIXTURE}" && -n "${CANDIDATE}" ]]; then
    echo ""
    python -m psych_asr.cli.check_split_regression "${FIXTURE}" "${CANDIDATE}"
else
    echo ""
    echo "Regression check SKIPPED: fixture='${FIXTURE}' candidate='${CANDIDATE}'"
fi

# The word-level diff across every arm that landed. Cheap, exact, and the thing to read
# before anyone opens a transcript: because Stage 1a ran once, the arms differ ONLY in the
# speaker label on each word, so this is a diff over one column rather than a reading task.
# Needs two arms to mean anything, so it is skipped rather than failed when only one landed.
if [[ "${JOINED}" -ge 2 ]]; then
    echo ""
    python -m psych_asr.cli.compare_arms || echo "Arm comparison failed (the per-arm transcripts are still valid)."
else
    echo ""
    echo "Arm comparison SKIPPED: only ${JOINED} arm(s) joined; a diff needs at least two."
fi

if [[ "${FAILED}" -ne 0 ]]; then
    exit 1
fi
