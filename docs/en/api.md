# API Reference

## supertonic_mnn.wrapper

### `SupertonicTTS`
```python
class SupertonicTTS(model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16")
```
High-level wrapper for the Text-to-Speech engine.

#### `synthesize`
```python
def synthesize(text: str, voice: str = "M1", steps: int = 5, speed: float = 1.0, output_file: Optional[str] = None) -> Tuple[np.ndarray, int]
```
Synthesizes speech from text.
*   `text`: Input text.
*   `voice`: Voice style name ("M1", "M2", "F1", "F2") or path to style JSON.
*   `steps`: Denoising steps (default 5).
*   `speed`: Speech speed (default 1.0).
*   `output_file`: If provided, saves the audio to this file.
*   Returns: `(audio_data, sample_rate)`

#### `save`
```python
@staticmethod
def save(filename: str, audio_data: np.ndarray, sample_rate: int)
```
Helper to save audio data to a file.

## supertonic_mnn.model

### `ensure_models`
```python
def ensure_models(target_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16")
```
Checks if models exist in `target_dir`. If not, downloads them from Hugging Face.

### `load_text_to_speech`
```python
def load_text_to_speech(model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16", use_gpu: bool = False) -> TextToSpeech
```
Initializes the TTS engine.

### `get_voice_style_path`
```python
def get_voice_style_path(voice_name: str, model_dir: str = DEFAULT_CACHE_DIR) -> str
```
Resolves the path for a given voice style name.

### `load_voice_style`
```python
def load_voice_style(voice_style_paths: list[str], verbose: bool = False) -> Style
```
Loads voice style vectors from JSON files.

## supertonic_mnn.engine

### `TextToSpeech`
The main inference class.

#### `__call__`
```python
def __call__(self, text: str, style: Style, total_step: int, speed: float = 1.05, silence_duration: float = 0.3) -> tuple[np.ndarray, np.ndarray, float]
```
Synthesizes speech from text.
