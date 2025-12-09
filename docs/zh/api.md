# API 参考手册

## supertonic_mnn.wrapper

### `SupertonicTTS`
```python
class SupertonicTTS(model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16")
```
文本转语音引擎的高级封装类。

#### `synthesize`
```python
def synthesize(text: str, voice: str = "M1", steps: int = 5, speed: float = 1.0, output_file: Optional[str] = None) -> Tuple[np.ndarray, int]
```
从文本合成语音。
*   `text`: 输入文本。
*   `voice`: 语音风格名称 ("M1", "M2", "F1", "F2") 或风格 JSON 文件路径。
*   `steps`: 去噪步数 (默认 5)。
*   `speed`: 语速 (默认 1.0)。
*   `output_file`: 如果提供，则将音频保存到此文件。
*   Returns: `(audio_data, sample_rate)`

#### `save`
```python
@staticmethod
def save(filename: str, audio_data: np.ndarray, sample_rate: int)
```
保存音频数据到文件的辅助方法。

## supertonic_mnn.model

### `ensure_models`
```python
def ensure_models(target_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16")
```
检查 `target_dir` 中是否存在模型。如果不存在，则从 Hugging Face 下载。

### `load_text_to_speech`
```python
def load_text_to_speech(model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16", use_gpu: bool = False) -> TextToSpeech
```
初始化 TTS 引擎。

### `get_voice_style_path`
```python
def get_voice_style_path(voice_name: str, model_dir: str = DEFAULT_CACHE_DIR) -> str
```
解析给定语音风格名称的路径。

### `load_voice_style`
```python
def load_voice_style(voice_style_paths: list[str], verbose: bool = False) -> Style
```
从 JSON 文件加载语音风格向量。

## supertonic_mnn.engine

### `TextToSpeech`
主要推理类。

#### `__call__`
```python
def __call__(self, text: str, style: Style, total_step: int, speed: float = 1.05, silence_duration: float = 0.3) -> tuple[np.ndarray, np.ndarray, float]
```
从文本合成语音。
