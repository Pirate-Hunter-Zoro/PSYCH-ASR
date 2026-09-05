"""Where the staged model weights and the pipeline's own artifacts live.

STDLIB ONLY. Every env imports this.

Model weights are staged ONCE on the login node into a models/ directory on study
storage and loaded thereafter by ABSOLUTE PATH with HF_HUB_OFFLINE=1, because the compute
nodes have no outbound internet (see README, "Staging model weights"). Those absolute
paths used to be repeated as literal defaults in five argument parsers, which meant
moving the staging directory was a five-file edit with no way to tell whether one had
been missed. They are here instead, and every parser's default reads from here.

The root is overridable with PSYCH_ASR_MODELS_ROOT for anyone running this off a
different filesystem; nothing in the pipeline sets it, so the default is what runs.
"""

import os
from pathlib import Path

# Study storage, not the home folder: the weights are ~5 GB and every compute node
# mounts this path.
MODELS_ROOT = Path(os.environ.get(
    "PSYCH_ASR_MODELS_ROOT",
    "/media/studies/ehr_study/analysis/mferguson/models",
))

# ---- Stage 1a: ASR and forced alignment ----
# Systran/faster-whisper-large-v3, staged by `hf download --local-dir`. Loaded with
# local_files_only, and the path must be handed over as a STRING -- only a str takes
# faster-whisper's local-directory branch.
WHISPER_MODEL_DIR = MODELS_ROOT / "faster-whisper-large-v3"

# ---- Stage 1b: one entry per arm ----
# Baseline. Pipeline.from_pretrained takes the DIRECTORY and finds config.yaml inside it;
# the config's $model/... references resolve against that same directory.
PYANNOTE_MODEL_DIR = MODELS_ROOT / "pyannote-speaker-diarization-community-1"

# Arm A. DiariZen's own from_pretrained calls snapshot_download, so the pipeline is
# constructed from these two absolute paths instead (see diarize/diarizen_arm.py).
DIARIZEN_MODEL_DIR = MODELS_ROOT / "diarizen-wavlm-large-s80-md-v2"
# The speaker embedder, handed over as a plain FILE path rather than a directory. Already
# staged for the baseline arm, so it is not a third download.
WESPEAKER_EMBEDDING_FILE = MODELS_ROOT / "pyannote-wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin"

# Arms B and C. Both load through SortformerEncLabelModel.restore_from, which reads the
# .nemo archive straight off disk and makes no Hub call at all.
SORTFORMER_OFFLINE_CHECKPOINT = MODELS_ROOT / "diar_sortformer_4spk-v1" / "diar_sortformer_4spk-v1.nemo"
SORTFORMER_STREAMING_CHECKPOINT = MODELS_ROOT / "diar_streaming_sortformer_4spk-v2.1" / "diar_streaming_sortformer_4spk-v2.1.nemo"

# ---- Caches that are NOT Hugging Face and are not covered by HF_HUB_OFFLINE ----
# WhisperX resolves its English alignment model to a torchaudio bundle fetched from
# download.pytorch.org, and its sentence splitter to an nltk package. Each library reads
# its OWN variable to find its cache, so every job must EXPORT these -- having the files
# on study storage is necessary and not sufficient.
TORCH_HOME = MODELS_ROOT / "torch_home"
NLTK_DATA = MODELS_ROOT / "nltk_data"

# ---- Artifacts ----
# Session content, and therefore PHI. data/ is gitignored wholesale; nothing here may be
# written anywhere else.
STAGE1_DIR = Path("data/stage1")
INBOX_DIR = Path("data/inbox")

# ---- Arm names, which are carried in every filename from Stage 1b onward ----
# Which model produced which transcript is a property of the file, not of a note
# somewhere, and Stage 1c derives the arm from the RTTM's own name -- so adding a fifth
# arm needs no change to the join.
ARM_BASELINE = "community-1"
ARM_DIARIZEN = "diarizen"
ARM_SORTFORMER = "sortformer"
ARM_SORTFORMER_STREAMING = "sortformer-streaming"
