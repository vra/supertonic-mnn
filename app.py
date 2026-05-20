"""Gradio demo for Supertonic MNN TTS."""

import os
import numpy as np
import gradio as gr

from supertonic_mnn.model import ensure_models, load_text_to_speech, get_voice_style_path, DEFAULT_CACHE_DIR
from supertonic_mnn.engine import load_voice_style
from supertonic_mnn.text import AVAILABLE_LANGS

LANG_NAMES = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "ar": "Arabic",
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "es": "Spanish", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "lt": "Lithuanian", "lv": "Latvian",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish",
    "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese",
    "na": "Language-Agnostic",
}

VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "converted_models")
USE_LOCAL = os.path.exists(LOCAL_MODEL_DIR)

tts_engines = {}
voice_style_cache = {}


def _load_engine_local(version: str, precision: str):
    from supertonic_mnn.engine import load_mnn, TextToSpeech
    from supertonic_mnn.text import UnicodeProcessor
    import json

    model_dir = os.path.join(LOCAL_MODEL_DIR, version)
    models_dir = os.path.join(model_dir, "mnn_models")
    precision_dir = os.path.join(models_dir, precision)
    cfg_path = os.path.join(models_dir, "tts.json")
    with open(cfg_path) as f:
        cfgs = json.load(f)
    mnn_cfg = {"backend": 0, "thread_num": 4, "precision": "low", "memory": "low"}
    dp = load_mnn(os.path.join(precision_dir, "duration_predictor.mnn"), ["text_ids", "style_dp", "text_mask"], ["duration"], mnn_cfg)
    text_enc = load_mnn(os.path.join(precision_dir, "text_encoder.mnn"), ["text_ids", "style_ttl", "text_mask"], ["text_emb"], mnn_cfg)
    vec_est = load_mnn(os.path.join(precision_dir, "vector_estimator.mnn"), ["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask", "current_step", "total_step"], ["denoised_latent"], mnn_cfg)
    vocoder = load_mnn(os.path.join(precision_dir, "vocoder.mnn"), ["latent"], ["wav_tts"], mnn_cfg)
    text_processor = UnicodeProcessor(os.path.join(models_dir, "unicode_indexer.json"))
    return TextToSpeech(cfgs, text_processor, dp, text_enc, vec_est, vocoder)


def get_engine(version: str, precision: str = "fp16"):
    key = f"{version}_{precision}"
    if key not in tts_engines:
        if USE_LOCAL:
            tts_engines[key] = _load_engine_local(version, precision)
        else:
            ensure_models(DEFAULT_CACHE_DIR, precision, version)
            tts_engines[key] = load_text_to_speech(DEFAULT_CACHE_DIR, precision, version=version)
    return tts_engines[key]


def get_style(voice: str, version: str):
    key = f"{version}_{voice}"
    if key not in voice_style_cache:
        if USE_LOCAL:
            style_path = os.path.join(LOCAL_MODEL_DIR, version, "voice_styles", f"{voice}.json")
        else:
            style_path = get_voice_style_path(voice, DEFAULT_CACHE_DIR, version)
        voice_style_cache[key] = load_voice_style([style_path])
    return voice_style_cache[key]


def synthesize(text: str, lang: str, voice: str, version: str, steps: int, speed: float):
    if not text.strip():
        return None, ""

    lang_code = lang.split(" - ")[0] if " - " in lang else lang

    engine = get_engine(version)
    style = get_style(voice, version)

    wav, duration, rtf = engine(text, lang_code, style, total_step=steps, speed=speed)

    audio_data = wav[0]
    sample_rate = engine.sample_rate
    audio_duration = audio_data.shape[0] / sample_rate

    info = (
        f"**RTF**: {rtf:.4f} | "
        f"**Audio Duration**: {audio_duration:.2f}s | "
        f"**Generation Time**: {audio_duration * rtf:.2f}s | "
        f"**Model**: {version}"
    )

    return (sample_rate, audio_data), info


lang_choices = [f"{code} - {LANG_NAMES.get(code, code)}" for code in AVAILABLE_LANGS]

with gr.Blocks(title="Supertonic MNN TTS") as demo:
    gr.Markdown("# Supertonic MNN TTS Demo")
    gr.Markdown("Text-to-Speech powered by MNN inference engine. Supports 30+ languages.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter text to synthesize...",
                lines=3,
                value="Hello! This is a demo of the Supertonic text to speech system running with MNN inference.",
            )
            lang_input = gr.Dropdown(
                choices=lang_choices,
                value="en - English",
                label="Language",
            )
            voice_input = gr.Dropdown(
                choices=VOICES,
                value="M1",
                label="Voice",
            )

        with gr.Column(scale=1):
            version_input = gr.Dropdown(
                choices=["v3", "v2"],
                value="v3",
                label="Model Version",
            )
            steps_input = gr.Slider(
                minimum=1, maximum=20, step=1, value=8,
                label="Diffusion Steps",
            )
            speed_input = gr.Slider(
                minimum=0.5, maximum=2.0, step=0.05, value=1.0,
                label="Speed",
            )

    synthesize_btn = gr.Button("Synthesize", variant="primary")
    audio_output = gr.Audio(label="Generated Audio", type="numpy")
    info_output = gr.Markdown(label="Info")

    synthesize_btn.click(
        fn=synthesize,
        inputs=[text_input, lang_input, voice_input, version_input, steps_input, speed_input],
        outputs=[audio_output, info_output],
    )

    gr.Markdown("---")
    gr.Markdown("### Examples")
    gr.Examples(
        examples=[
            ["Hello! This is a demo of the Supertonic text to speech system.", "en - English", "M1", "v3", 8, 1.0],
            ["안녕하세요! 수퍼토닉 음성 합성 시스템의 데모입니다.", "ko - Korean", "F1", "v3", 8, 1.0],
            ["こんにちは！スーパートニック音声合成システムのデモです。", "ja - Japanese", "M2", "v3", 8, 1.0],
            ["Bonjour! Ceci est une démonstration du système de synthèse vocale Supertonic.", "fr - French", "F2", "v3", 8, 1.0],
            ["Hallo! Dies ist eine Demo des Supertonic Text-to-Speech-Systems.", "de - German", "M3", "v3", 8, 1.0],
        ],
        inputs=[text_input, lang_input, voice_input, version_input, steps_input, speed_input],
    )


if __name__ == "__main__":
    demo.launch()
