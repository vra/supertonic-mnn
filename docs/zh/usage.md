# 使用指南

## Python API

您可以在 Python 项目中直接调用 `supertonic-mnn`。

### 基础示例

简化后的 API 封装了模型加载和推理过程。

```python
from supertonic_mnn import SupertonicTTS

# 1. 初始化
# 精度可以是 "fp16" (默认), "fp32", 或 "int8"
tts = SupertonicTTS(precision="fp16")

# 2. 合成
# 如果默认缓存目录中没有模型，会自动下载。
# 您可以指定 output_file 参数直接保存音频。
audio_data, sample_rate = tts.synthesize(
    text="欢迎使用 Supertonic MNN。",
    voice="M1",
    output_file="output.wav"
)

print(f"语音已合成，采样率为 {sample_rate}Hz。")
```

### 进阶用法

您可以自定义模型目录，并按需使用底层 API。

```python
from supertonic_mnn import SupertonicTTS

# 指定自定义模型目录和精度
tts = SupertonicTTS(model_dir="./custom_models", precision="int8")

# 合成但不保存到文件 (返回 numpy 数组)
audio_data, sample_rate = tts.synthesize("你好，世界", voice="F1", speed=1.2)

# 使用辅助方法手动保存
tts.save("fast_hello.wav", audio_data, sample_rate)
```

## 命令行接口 (CLI)

本软件包提供了一个 `supertonic-mnn` 命令，用于快速使用。

### 文本合成

```bash
supertonic-mnn -i input.txt -o output.wav --voice M1
```

或者使用标准输入：

```bash
echo "你好，世界" | supertonic-mnn -o hello.wav
```

### 可用音色
| 音色 ID | 描述 |
| :--- | :--- |
| **M1** | 男声 1 |
| **M2** | 男声 2 |
| **F1** | 女声 1 |
| **F2** | 女声 2 |

### 选项

*   `-i, --input-file`: 文本文件路径 (utf-8)。
*   `-o, --output`: 输出 wav 文件路径。
*   `--voice`: 语音风格名称 (M1, M2, F1, F2) 或风格 JSON 文件路径。
*   `--speed`: 语速 (默认 1.0)。
*   `--steps`: 扩散步数 (默认 5)。
*   `--precision`: 模型精度 (fp16, fp32, int8)。
