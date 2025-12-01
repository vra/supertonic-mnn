import argparse
import soundfile as sf
import re
from .engine import load_voice_style
from .model import (
    ensure_models,
    load_text_to_speech,
    get_voice_style_path,
    DEFAULT_CACHE_DIR,
)


def sanitize_filename(text: str, max_len: int = 20) -> str:
    """Sanitize filename by replacing non-alphanumeric characters with underscores"""
    prefix = text[:max_len]
    return re.sub(r"[^a-zA-Z0-9]", "_", prefix)


def main():
    parser = argparse.ArgumentParser(description="Supertonic MNN Inference CLI")

    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        help="Text to synthesize. If not provided, reads from stdin.",
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

    args = parser.parse_args()

    # Handle input text
    if args.text:
        text = args.text
    else:
        import sys

        print("Reading text from stdin...")
        text = sys.stdin.read().strip()

    if not text:
        print("Error: No text provided.")
        return

    # 1. Ensure models are present
    try:
        ensure_models(args.model_dir, args.precision)
    except Exception as e:
        print(f"Error downloading models: {e}")
        return

    # 2. Load TTS Engine
    print(f"Loading TTS engine with precision={args.precision}...")
    try:
        tts = load_text_to_speech(args.model_dir, args.precision)
    except Exception as e:
        print(f"Error loading engine: {e}")
        return

    # 3. Load Voice Style
    try:
        style_path = get_voice_style_path(args.voice, args.model_dir)
        print(f"Using voice style: {style_path}")
        style = load_voice_style([style_path])
    except Exception as e:
        print(f"Error loading voice style: {e}")
        return

    # 4. Synthesize
    print(f"Synthesizing text: '{text[:50]}...'")
    try:
        wav, duration = tts(text, style, args.steps, args.speed)

        # 5. Save output
        # wav is [1, T]
        wav_data = wav[0]
        sf.write(args.output, wav_data, tts.sample_rate)
        print(f"Saved audio to: {args.output}")

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
