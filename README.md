# Supertonic MNN CLI

A command-line interface for running Supertonic TTS models using MNN.

## Installation

```bash
cd supertonic-mnn
uv sync
```

## Usage

```bash
# Using local models with default precision (fp16)
uv run supertonic-mnn "Hello world" --voice M1 --output hello.wav --model-dir /path/to/models

# Specify precision
uv run supertonic-mnn "Hello world" --voice M1 --precision fp32

# Download models from HuggingFace (automatic)
uv run supertonic-mnn "Hello world" --voice M1 --precision int8

# Example with py directory (if you have local models in old format)
uv run supertonic-mnn "Hello world" --voice M1 --output hello.wav --model-dir ../py
```

### Available Options

- `--voice`: Voice style (default: M1)
- `--precision`: Model precision - fp32, fp16, or int8 (default: fp16)
- `--output`: Output audio file path (default: output.wav)
- `--speed`: Speech speed multiplier (default: 1.0)
- `--steps`: Number of denoising steps (default: 5)
- `--model-dir`: Directory containing models (default: ~/.cache/supertonic-mnn)

## Features

- **MNN Inference**: Fast, on-device inference using MNN
- **Voice Styles**: Supports multiple voice styles (M1, F1, etc.)
- **Local Models**: Use `--model-dir` to specify local model directory
