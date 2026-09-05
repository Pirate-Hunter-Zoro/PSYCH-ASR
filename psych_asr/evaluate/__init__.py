"""Comparing the arms with no reference, and scoring them against one.

compare and regression are stdlib only. score needs pyannote.metrics and therefore
diar_eval_env, which has no torch and must keep it that way.

Empty on purpose -- see psych_asr/__init__.py for why nothing is re-exported.
"""
