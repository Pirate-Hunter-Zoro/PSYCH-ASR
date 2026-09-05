"""PSYCH-ASR pipeline package: ASR, diarization, the join, and the bake-off scorers.

DELIBERATELY EMPTY OF IMPORTS, AND EVERY SUBPACKAGE'S __init__.py IS TOO.

This package is imported from FOUR conda envs whose torch, numpy and transformers pins
are mutually incompatible (asr_env, diarizen_env, nemo_env, diar_eval_env). A convenience
re-export here -- `from psych_asr import diarize_pyannote` and friends -- would make
importing ANY module drag in EVERY module's third-party dependencies, and diar_eval_env
would stop being able to read an RTTM without a torch install.

So: nothing is re-exported anywhere. Callers name the module they want in full, and each
module declares only what it actually needs. The import graph is the env contract.

Nothing here is installed into any env. Every entry point runs as `python -m
psych_asr.cli.<name>` FROM THE REPO ROOT, which is what puts the repo on sys.path.
"""

__version__ = "0.2.0"
