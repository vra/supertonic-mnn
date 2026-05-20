import argparse
import soundfile as sf
import re
import os
import sys
from .engine import load_voice_style
from .model import (
    ensure_models,
    load_text_to_speech,
    get_voice_style_path,
    DEFAULT_CACHE_DIR,
)


def sanitize_filename(text: str, max_len: int = 20) -> str:
    """Sanitize filename by replacing non-alphanumeric characters with underscores (supports Unicode)"""
    prefix = text[:max_len]
    return re.sub(r"[^\w]", "_", prefix, flags=re.UNICODE)


def main():
    parser = argparse.ArgumentParser(description="Supertonic MNN Inference CLI")

    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="Input text file to synthesize. Each line will be synthesized separately. "
             "If not provided, reads from stdin.",
    )

    parser.add_argument(
        "--voice",
        type=str,
        default="M1",
        help="Voice style name (e.g., M1, Z1) or path to style JSON file. Default: M1",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output audio file path. Default: output.wav",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (default: 1.0, higher = faster)",
    )

    parser.add_argument(
        "--steps", type=int, default=5, help="Number of denoising steps (default: 5)"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory to store/load models. Default: {DEFAULT_CACHE_DIR}",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Model precision: fp32, fp16, or int8. Default: fp16",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code (e.g., en, ko, ja, fr, de). Default: en",
    )

    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2", "v3"],
        default="v3",
        help="Model version: v1, v2, or v3. Default: v3",
    )

    args = parser.parse_args()

    # Handle input text
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file '{args.input_file}' does not exist.")
            return
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            texts = [line.strip() for line in lines if line.strip()]
        if not texts:
            print(f"Error: No valid text found in '{args.input_file}'.")
            return
        print(f"Loaded {len(texts)} line(s) from '{args.input_file}'.")
    else:
        print("Reading text from stdin...")
        text = sys.stdin.read().strip()
        if not text:
            print("Error: No text provided.")
            return
        texts = [text]

    if not texts:
        print("Error: No text provided.")
        return

    # 1. Ensure models are present
    try:
        ensure_models(args.model_dir, args.precision, args.version)
    except Exception as e:
        print(f"Error downloading models: {e}")
        return

    # 2. Load TTS Engine
    print(f"Loading TTS engine with precision={args.precision}, version={args.version}...")
    try:
        tts = load_text_to_speech(args.model_dir, args.precision, version=args.version)
    except Exception as e:
        print(f"Error loading engine: {e}")
        return

    # 3. Load Voice Style
    try:
        style_path = get_voice_style_path(args.voice, args.model_dir, args.version)
        print(f"Using voice style: {style_path}")
        style = load_voice_style([style_path])
    except Exception as e:
        print(f"Error loading voice style: {e}")
        return

    # 4. Synthesize
    if args.input_file and len(texts) > 1:
        print(f"Synthesizing {len(texts)} line(s) from file...")
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
        base_name = os.path.splitext(os.path.basename(args.output))[0]
        extension = os.path.splitext(args.output)[1] if os.path.splitext(args.output)[1] else ".wav"
        
        for idx, text in enumerate(texts, 1):
            print(f"Synthesizing line {idx}/{len(texts)}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            try:
                wav, duration, rtf = tts(text, args.lang, style, args.steps, args.speed)
                
                # Generate output filename
                if len(texts) == 1:
                    output_file = args.output
                else:
                    output_file = os.path.join(output_dir, f"{base_name}_{idx}{extension}")
                
                # Save output
                wav_data = wav[0]
                sf.write(output_file, wav_data, tts.sample_rate)
                print(f"Saved audio to: {output_file}")
                
            except Exception as e:
                print(f"Inference failed for line {idx}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Single text synthesis
        text = texts[0]
        print(f"Synthesizing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        try:
            wav, duration, rtf = tts(text, args.lang, style, args.steps, args.speed)

            # Save output
            wav_data = wav[0]
            sf.write(args.output, wav_data, tts.sample_rate)
            print(f"Saved audio to: {args.output}")

        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()