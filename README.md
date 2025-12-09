# Supertonic MNN CLI
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/yunfengwang/supertonic-tts-mnn)

A command-line interface for running [Supertonic TTS models](https://github.com/supertone-inc/supertonic) using [MNN](https://github.com/alibaba/MNN).

## Features
- **MNN Inference**: Fast, on-device inference using MNN, **RTF ~ 0.07**
- **Int8 Supports**: no loss of precisions compared with fp32 and fp16

## Usage
Install by pip and run:
```bash
pip install supertonic-mnn
# Provide text through stdin
echo "Hello world" | supertonic-mnn --output out.wav

# Or read from a text file
supertonic-mnn --input-file sentences.txt --voice F1 --precision int8 --output out.wav
```

### Available Options

- `--input-file`, `-i`: Input text file to synthesize (each line will be synthesized separately)
- `--voice`: Voice style (default: M1, choices: M1, M2, F1, F2)
- `--precision`: Model precision - fp32, fp16, or int8 (default: fp16)
- `--output`, `-o`: Output audio file path (default: output.wav)
- `--speed`: Speech speed multiplier (default: 1.0)
- `--steps`: Number of denoising steps (default: 5)
- `--model-dir`: Directory containing models


## Installation By Source Code
```bash
git clone https://github.com/vra/supertonic-mnn
cd supertonic-mnn
uv sync
```

## Usage

```bash
# Reading text from stdin
echo "Hello world" | supertonic-mnn --output hello.wav

# Using local models with default precision (fp16)
echo "Hello world" | supertonic-mnn --output hello.wav --model-dir /path/to/models

# Specify precision
echo "Hello world" | supertonic-mnn --output hello.wav --precision fp32

# Download models from HuggingFace (automatic)
echo "Hello world" | supertonic-mnn --output hello.wav --precision int8

# Batch processing from text file
uv run supertonic-mnn --input-file sentences.txt --voice F1 --output result.wav
```
