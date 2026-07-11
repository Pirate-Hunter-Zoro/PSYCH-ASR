import sys
import torch
from faster_whisper import WhisperModel

wav_path = sys.argv[1]

print(torch.__version__)
print(torch.cuda.get_device_name(device=0))
    
whisper_model = WhisperModel("/media/studies/ehr_study/analysis/mferguson/models/faster-whisper-tiny", device="cuda", compute_type="float16")
segments, info = whisper_model.transcribe(beam_size=5, audio=wav_path)
print(info.language)
segments = list(segments)
for s in segments:
    print(s.text)
print("SMOKE OK")