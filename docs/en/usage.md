# Usage Guide

## Python API

You can use `supertonic-mnn` directly in your Python projects.

### Basic Example

The simplified API encapsulates model loading and inference.

```python
from supertonic_mnn import SupertonicTTS

# 1. Initialize
# Precision can be "fp16" (default), "fp32", or "int8"
tts = SupertonicTTS(precision="fp16")

# 2. Synthesize
# Models will be downloaded automatically if not present in the default cache directory.
# You can specify the output file name to save the audio directly.
audio_data, sample_rate = tts.synthesize(
    text="Welcome to Supertonic MNN.",
    voice="M1",
    output_file="output.wav"
)

print(f"Audio synthesized with sample rate {sample_rate}Hz.")
```

### Advanced Usage

You can customize the model directory and access the underlying engine if needed.

```python
from supertonic_mnn import SupertonicTTS

# Initialize with custom model directory and precision
tts = SupertonicTTS(model_dir="./custom_models", precision="int8")

# Synthesize without saving to file (returns numpy array)
audio_data, sample_rate = tts.synthesize("Hello world", voice="F1", speed=1.2)

# Save manually using the helper method
tts.save("fast_hello.wav", audio_data, sample_rate)
```

## Command Line Interface (CLI)

The package provides a `supertonic-mnn` command for quick usage.

### Synthesize Text

```bash
supertonic-mnn -i input.txt -o output.wav --voice M1
```

Or using stdin:

```bash
echo "Hello World" | supertonic-mnn -o hello.wav
```

### Available Voices

| Voice ID | Description |
| :--- | :--- |
| **M1** | Male voice 1 |
| **M2** | Male voice 2 |
| **F1** | Female voice 1 |
| **F2** | Female voice 2 |

### Options

*   `-i, --input-file`: Path to text file (utf-8).
*   `-o, --output`: Output wav file path.
*   `--voice`: Voice style (M1, M2, F1, F2) or path to style json.
*   `--speed`: Speech speed (default 1.0).
*   `--steps`: Diffusion steps (default 5).
*   `--precision`: Model precision (fp16, fp32, int8).
