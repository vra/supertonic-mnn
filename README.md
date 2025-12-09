# Supertonic MNN

[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/yunfengwang/supertonic-tts-mnn)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://vra.github.io/supertonic-mnn/)
[![PyPI](https://img.shields.io/pypi/v/supertonic-mnn)](https://pypi.org/project/supertonic-mnn/)

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**Supertonic MNN** is a high-performance, lightweight text-to-speech (TTS) library based on MNN. It supports both command-line interface (CLI) and Python API, making it easy to integrate into your projects.

### Features
*   **Fast Inference**: RTF ~ 0.07 on CPU.
*   **Lightweight**: Minimal dependencies.
*   **Supported Precisions**: fp32, fp16, int8.

### Available Voices
*   **M1**: Male voice 1
*   **M2**: Male voice 2
*   **F1**: Female voice 1
*   **F2**: Female voice 2

### Documentation
Full documentation is available at [Supertonic MNN Docs](https://vra.github.io/supertonic-mnn/).

### Installation

```bash
pip install supertonic-mnn
```

### Quick Usage

#### CLI
```bash
echo "Hello world" | supertonic-mnn -o hello.wav
```

#### Python API
```python
from supertonic_mnn import SupertonicTTS

# 1. Initialize
tts = SupertonicTTS()

# 2. Synthesize
# Models will be downloaded automatically if not present
audio, sample_rate = tts.synthesize("Hello world", voice="M1", output_file="hello.wav")
```

See [examples/](examples/) for more details.

### Acknowledgments
This project is based on the original [Supertonic](https://github.com/supertone-inc/supertonic/) by Supertone Inc.

---

<a name="中文"></a>
## 中文

**Supertonic MNN** 是一个基于 MNN 的高性能、轻量级文本转语音 (TTS) 库。它同时支持命令行接口 (CLI) 和 Python API，方便您将其集成到项目中。

### 特性
*   **极速推理**: CPU 上 RTF 约为 0.07。
*   **轻量级**: 依赖极少。
*   **多精度支持**: fp32, fp16, int8。

### 可用音色
*   **M1**: 男声 1
*   **M2**: 男声 2
*   **F1**: 女声 1
*   **F2**: 女声 2

### 文档
完整文档请访问 [Supertonic MNN 文档](https://vra.github.io/supertonic-mnn/)。

### 安装

```bash
pip install supertonic-mnn
```

### 快速上手

#### 命令行 (CLI)
```bash
echo "你好，世界" | supertonic-mnn -o hello.wav
```

#### Python API
```python
from supertonic_mnn import SupertonicTTS

# 1. 初始化
tts = SupertonicTTS()

# 2. 推理
# 如果模型不存在，会自动下载
audio, sample_rate = tts.synthesize("你好，世界", voice="M1", output_file="hello.wav")
```

更多详情请参阅 [examples/](examples/) 目录。

### 致谢
本项目基于 Supertone Inc. 的原始 [Supertonic](https://github.com/supertone-inc/supertonic/) 工作。
