import torchaudio
import os

pipeline_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = pipeline_bundle.get_model()
labels = pipeline_bundle.get_labels()

print(f"Sample rate: {pipeline_bundle.sample_rate}\nNumber of labels: {len(labels)}\nTORCH_HOME: {os.environ.get('TORCH_HOME', 'unset')}", flush=True)