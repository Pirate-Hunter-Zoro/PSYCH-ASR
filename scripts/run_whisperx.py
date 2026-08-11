from argparse import ArgumentParser
from pathlib import Path
import json
import whisperx

def main():
    parser = ArgumentParser()
    parser.add_argument("audio", type=str)
    parser.add_argument("--outdir", type=str, default="data/stage1")
    parser.add_argument("--model-dir", type=str, default="/media/studies/ehr_study/analysis/mferguson/models/faster-whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    audio_path = Path(args.audio)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded_audio = whisperx.load_audio(audio_path)
    pipeline_model = whisperx.load_model(args.model_dir, device="cuda", compute_type="float16", language="en", local_files_only=True)
    processed_audio = pipeline_model.transcribe(decoded_audio, batch_size=args.batch_size, print_progress=True)
    output_path = output_dir / (audio_path.stem + ".asr.json")
    with open(output_path, 'w') as f:
        json.dump(processed_audio, f, indent=4, ensure_ascii=False)
        
    print(f"Number of segments: {len(processed_audio['segments'])}\nLast segment: {processed_audio['segments'][-1]['end']}\nDetected language: {processed_audio['language']}", flush=True)

if __name__=="__main__":
    main()