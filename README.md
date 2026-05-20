# Supertonic MNN

[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/yunfengwang/supertonic-tts-mnn)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://vra.github.io/supertonic-mnn/)
[![PyPI](https://img.shields.io/pypi/v/supertonic-mnn)](https://pypi.org/project/supertonic-mnn/)

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**Supertonic MNN** is a high-performance, lightweight multilingual text-to-speech (TTS) library powered by [MNN](https://github.com/alibaba/MNN) inference engine. It supports 30+ languages, 10 voice styles, and provides both CLI and Python API.

Demo video: <https://www.bilibili.com/video/BV1VFqiBSER3>

### Features

- **Multilingual**: 30+ languages (English, Korean, Japanese, French, German, Spanish, etc.)
- **Fast Inference**: RTF ~ 0.07 on CPU
- **Multiple Model Versions**: v2 (multilingual) and v3 (30+ languages)
- **Precision Options**: fp32, fp16, int8
- **10 Voice Styles**: M1-M5 (male), F1-F5 (female)

### Models

| Version | Languages | HuggingFace |
|---------|-----------|-------------|
| v3 | 30+ languages + language-agnostic | [yunfengwang/supertonic-tts-mnn](https://huggingface.co/yunfengwang/supertonic-tts-mnn) |
| v2 | en, ko, es, pt, fr | [yunfengwang/supertonic-tts-mnn](https://huggingface.co/yunfengwang/supertonic-tts-mnn) |

Original ONNX models: [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2), [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)

### Installation

Requires Python 3.10 (MNN constraint).

```bash
# Install with uv (recommended)
uv sync

# Or install with pip
pip install supertonic-mnn
```

To run the Gradio demo:

```bash
uv sync --group demo
uv run python app.py
```

### Quick Usage

#### CLI

```bash
# English (default)
echo "Hello world" | uv run supertonic-mnn -o hello.wav

# Korean
echo "안녕하세요" | uv run supertonic-mnn --lang ko -o hello_ko.wav

# Japanese with v3 model, female voice
echo "こんにちは" | uv run supertonic-mnn --lang ja --version v3 --voice F1 -o hello_ja.wav
```

#### Python API

```python
from supertonic_mnn import SupertonicTTS

tts = SupertonicTTS(version="v3")

# English
audio, sr = tts.synthesize("Hello world", lang="en", voice="M1", output_file="hello.wav")

# Korean
audio, sr = tts.synthesize("안녕하세요", lang="ko", voice="F1", output_file="hello_ko.wav")

# Japanese
audio, sr = tts.synthesize("こんにちは", lang="ja", voice="M2", output_file="hello_ja.wav")
```

### Supported Languages

`en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`, `na` (language-agnostic)

### Documentation

Full documentation: [Supertonic MNN Docs](https://vra.github.io/supertonic-mnn/)

### Acknowledgments

This project is based on the original [Supertonic](https://github.com/supertone-inc/supertonic/) by Supertone Inc.

---

<a name="中文"></a>
## 中文

**Supertonic MNN** 是一个基于 [MNN](https://github.com/alibaba/MNN) 推理引擎的高性能、轻量级多语言文本转语音 (TTS) 库。支持 30+ 语言、10 种音色，同时提供命令行和 Python API。

Demo video: <https://www.bilibili.com/video/BV1VFqiBSER3>

### 特性

- **多语言支持**: 30+ 语言（英语、韩语、日语、法语、德语、西班牙语等）
- **极速推理**: CPU 上 RTF 约 0.07
- **多模型版本**: v2（多语言）和 v3（30+ 语言）
- **多精度**: fp32, fp16, int8
- **10 种音色**: M1-M5（男声），F1-F5（女声）

### 模型

| 版本 | 语言 | HuggingFace |
|------|------|-------------|
| v3 | 30+ 语言 + 语言无关模式 | [yunfengwang/supertonic-tts-mnn](https://huggingface.co/yunfengwang/supertonic-tts-mnn) |
| v2 | en, ko, es, pt, fr | [yunfengwang/supertonic-tts-mnn](https://huggingface.co/yunfengwang/supertonic-tts-mnn) |

原始 ONNX 模型: [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2), [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)

### 安装

需要 Python 3.10（MNN 限制）。

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip 安装
pip install supertonic-mnn
```

运行 Gradio demo:

```bash
uv sync --group demo
uv run python app.py
```

### 快速上手

#### 命令行 (CLI)

```bash
# 英语（默认）
echo "Hello world" | uv run supertonic-mnn -o hello.wav

# 韩语
echo "안녕하세요" | uv run supertonic-mnn --lang ko -o hello_ko.wav

# 日语，v3 模型，女声
echo "こんにちは" | uv run supertonic-mnn --lang ja --version v3 --voice F1 -o hello_ja.wav
```

#### Python API

```python
from supertonic_mnn import SupertonicTTS

tts = SupertonicTTS(version="v3")

# 英语
audio, sr = tts.synthesize("Hello world", lang="en", voice="M1", output_file="hello.wav")

# 韩语
audio, sr = tts.synthesize("안녕하세요", lang="ko", voice="F1", output_file="hello_ko.wav")

# 日语
audio, sr = tts.synthesize("こんにちは", lang="ja", voice="M2", output_file="hello_ja.wav")
```

### 支持的语言

`en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`, `na`（语言无关）

### 文档

完整文档: [Supertonic MNN Docs](https://vra.github.io/supertonic-mnn/)

### 致谢

本项目基于 Supertone Inc. 的原始 [Supertonic](https://github.com/supertone-inc/supertonic/) 工作。
