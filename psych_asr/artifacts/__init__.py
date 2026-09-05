"""On-disk artifact shapes: RTTM, the aligned/diarized JSON, and the naming convention.

STDLIB ONLY, at module scope, throughout this subpackage. It is the one part of the
package that every env imports, including torch-free diar_eval_env. pandas is imported
inside the single function that returns a DataFrame and nowhere else.

Empty on purpose -- see psych_asr/__init__.py for why nothing is re-exported.
"""
