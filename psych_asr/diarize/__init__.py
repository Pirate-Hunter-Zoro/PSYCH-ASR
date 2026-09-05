"""Stage 1b: one module per bake-off arm, each importable only in its own env.

pyannote_arm needs asr_env, diarizen_arm needs diarizen_env, sortformer_arm needs
nemo_env. Their torch pins conflict, so this file re-exporting any of them would make
every arm unimportable everywhere except where all three could coexist -- which is
nowhere. windowing is numpy/scipy only and is shared.

Empty on purpose -- see psych_asr/__init__.py for why nothing is re-exported.
"""
