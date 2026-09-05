"""Fetch the torchaudio forced-alignment bundle into TORCH_HOME. Login node only.

    TORCH_HOME=<models>/torch_home python -m psych_asr.cli.warm_align_cache

FORCED ALIGNMENT IS NOT A HUGGING FACE DOWNLOAD. For English, WhisperX resolves its
alignment model to the TORCHAUDIO bundle WAV2VEC2_ASR_BASE_960H, fetched from
download.pytorch.org into the Torch hub cache -- not from the Hugging Face Hub. Setting
HF_HUB_OFFLINE=1 therefore does NOT protect this path, and on a compute node with no
outbound internet it stalls exactly the way an unstaged Hub model does.

torch writes the 360 MB wav2vec2_fairseq_base_ls960_asr_ls960.pth into a hub/checkpoints
subfolder of TORCH_HOME, and a re-run reuses it silently. EVERY JOB MUST EXPORT THE SAME
TORCH_HOME -- exporting the variable, not merely having the files on disk, is what makes
torch reuse the cache instead of re-fetching into ~/.cache.
"""

import os

import torchaudio


def main(argv=None):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    bundle.get_model()
    labels = bundle.get_labels()
    print(
        f"Sample rate: {bundle.sample_rate}\n"
        f"Number of labels: {len(labels)}\n"
        f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'unset')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
