import sys
import os

# Add src to path if running from root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from supertonic_mnn import SupertonicTTS

def main():
    # Initialize the engine (automatic model download if needed)
    tts = SupertonicTTS()

    # Synthesize and save
    print("Synthesizing...")
    audio, sample_rate = tts.synthesize(
        text="Hello, this is the simplifed Supertonic MNN API.",
        voice="M1",
        output_file="output_simple.wav"
    )
    print("Done! Saved to output_simple.wav")

if __name__ == "__main__":
    main()
