#!/bin/bash
# ---------------------------------------------------------------------------
# standardize.sh -- Stage 0: one recording in, a 16 kHz mono WAV out.
#
#     bash scripts/standardize.sh data/<recording>.m4a
#
# The pilot recordings are ~50 min, 32 kHz, mono AAC -- a SINGLE MIXED CHANNEL,
# so diarization cannot lean on channel separation. 16 kHz mono is what every
# model downstream expects: Whisper resamples to it, and pyannote's segmentation
# and embedding models are trained at it.
#
# The WAV is written BESIDE the source recording. Moving the one you want
# processed into data/inbox/ is a deliberate manual step -- the Stage 1 jobs take
# no arguments and glob that directory, so what gets processed is decided by
# where the file is, not by a flag someone might mistype.
#
# Output is PHI. data/ is gitignored wholesale; do not write it anywhere else.
# ---------------------------------------------------------------------------

set -o errexit
set -o nounset
set -o pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/standardize.sh <recording>" >&2
    exit 1
fi

input="$1"
if [ ! -f "$input" ]; then
    echo "no such file: $input" >&2
    exit 1
fi

# Replace whatever extension it has, rather than assuming .m4a: the pilot is AAC
# but a collaborator's overlap recording may not be.
output="${input%.*}.wav"
if [ "$output" = "$input" ]; then
    echo "refusing to overwrite $input with itself -- it is already a .wav" >&2
    exit 1
fi

#  -ar 16000  resample to 16 kHz      -ac 1  force mono
#  -c:a pcm_s16le  uncompressed 16-bit PCM, which is what soundfile and torchaudio read
ffmpeg -y -i "$input" -ar 16000 -ac 1 -c:a pcm_s16le "$output"

echo "wrote $output"
