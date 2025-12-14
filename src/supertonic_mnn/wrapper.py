import os
import soundfile as sf
import numpy as np
from typing import Optional, Union, Tuple
from .model import ensure_models, load_text_to_speech, get_voice_style_path, DEFAULT_CACHE_DIR
from .engine import load_voice_style

class SupertonicTTS:
    """
    A high-level wrapper for Supertonic MNN Text-to-Speech Engine.
    
    Usage:
        tts = SupertonicTTS()
        audio, sample_rate = tts.synthesize("Hello world")
        tts.save("output.wav", audio, sample_rate)
    """

    def __init__(self, model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16"):
        """
        Initialize the TTS engine.
        
        Args:
            model_dir (str): Directory to store/load models.
            precision (str): Model precision ('fp16', 'fp32', 'int8').
        """
        self.model_dir = model_dir
        self.precision = precision
        self.engine = None
        self.voice_styles = {}
        
        # Ensure models are available upon initialization
        ensure_models(self.model_dir, self.precision)

    def _get_engine(self):
        """Lazily loads the TTS engine."""
        if self.engine is None:
            self.engine = load_text_to_speech(self.model_dir, self.precision)
        return self.engine

    def synthesize(
        self, 
        text: str, 
        voice: str = "M1", 
        steps: int = 5, 
        speed: float = 1.0, 
        output_file: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize text to speech.

        Args:
            text (str): Text to synthesize.
            voice (str): Voice style name (e.g., 'M1', 'F1') or path to style JSON.
            steps (int): Number of diffusion steps (default 5).
            speed (float): Speech speed (default 1.0).
            output_file (str, optional): Path to save the output audio file.

        Returns:
            (audio_data, sample_rate): Numpy array of audio data and sample rate.
        """
        engine = self._get_engine()
        
        # Load or retrieve voice style
        if voice not in self.voice_styles:
            style_path = get_voice_style_path(voice, self.model_dir)
            self.voice_styles[voice] = load_voice_style([style_path])
        
        style = self.voice_styles[voice]
        
        wav, duration, rtf = engine(text, style, total_step=steps, speed=speed)
        
        wav_data = wav[0]
        sample_rate = engine.sample_rate
        
        if output_file:
            self.save(output_file, wav_data, sample_rate)
            
        return wav_data, sample_rate

    def synthesize_stream(
        self, 
        text: str, 
        voice: str = "M1", 
        steps: int = 5, 
        speed: float = 1.0, 
    ):
        """
        Synthesize text to speech as a stream (generator).

        Args:
            text (str): Text to synthesize.
            voice (str): Voice style name.
            steps (int): Number of diffusion steps.
            speed (float): Speech speed.

        Yields:
            (audio_chunk, sample_rate): Tuple of audio chunk (numpy array) and sample rate.
        """
        engine = self._get_engine()
        
        # Load or retrieve voice style
        if voice not in self.voice_styles:
            style_path = get_voice_style_path(voice, self.model_dir)
            self.voice_styles[voice] = load_voice_style([style_path])
        
        style = self.voice_styles[voice]
        
        stream_gen = engine.stream(text, style, total_step=steps, speed=speed)
        
        sample_rate = engine.sample_rate
        
        for wav, duration, elapsed in stream_gen:
            yield wav[0], sample_rate

    @staticmethod
    def save(filename: str, audio_data: np.ndarray, sample_rate: int):
        """
        Save audio data to a file.
        
        Args:
            filename (str): Output file path.
            audio_data (np.ndarray): Audio data.
            sample_rate (int): Sample rate.
        """
        sf.write(filename, audio_data, sample_rate)
