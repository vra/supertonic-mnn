import os
import json
import time
from huggingface_hub import hf_hub_download
from .engine import TextToSpeech, load_mnn
from .text import UnicodeProcessor

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/supertonic-mnn")
REPO_ID = "yunfengwang/supertonic-tts-mnn"


def ensure_models(target_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16"):
    """
    Ensure that the MNN models and voice styles are present in the target directory.
    If not, download them from Hugging Face.
    """
    print(f"Checking models in {target_dir}...")

    # Check if files already exist
    models_dir = os.path.join(target_dir, "mnn_models")
    precision_dir = os.path.join(models_dir, precision)

    required_files = [
        os.path.join(target_dir, "config.json"),
        os.path.join(precision_dir, "duration_predictor.mnn"),
        os.path.join(precision_dir, "text_encoder.mnn"),
        os.path.join(precision_dir, "vector_estimator.mnn"),
        os.path.join(precision_dir, "vocoder.mnn"),
        os.path.join(models_dir, "tts.json"),
        os.path.join(models_dir, "unicode_indexer.json"),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if not missing_files:
        print(f"All required models found in {target_dir}")
        return

    print(f"Missing files: {missing_files}")
    print(f"Attempting to download from HF ({REPO_ID}) with precision={precision}...")

    # Download config files if missing
    try:
        if not os.path.exists(os.path.join(target_dir, "config.json")):
            print("Downloading config.json...")
            hf_hub_download(
                repo_id=REPO_ID, filename="config.json", local_dir=target_dir
            )

        if not os.path.exists(os.path.join(models_dir, "tts.json")):
            print("Downloading tts.json...")
            hf_hub_download(
                repo_id=REPO_ID, filename="mnn_models/tts.json", local_dir=target_dir
            )

        if not os.path.exists(os.path.join(models_dir, "unicode_indexer.json")):
            print("Downloading unicode_indexer.json...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename="mnn_models/unicode_indexer.json",
                local_dir=target_dir
            )
    except Exception as e:
        print(f"Failed to download config files: {e}")
        if missing_files:
            raise RuntimeError(
                f"Models missing in {target_dir} and download failed."
            ) from e

    # Download precision-specific model files
    model_files = [
        f"mnn_models/{precision}/duration_predictor.mnn",
        f"mnn_models/{precision}/text_encoder.mnn",
        f"mnn_models/{precision}/vector_estimator.mnn",
        f"mnn_models/{precision}/vocoder.mnn",
    ]

    for filename in model_files:
        local_path = os.path.join(target_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID, filename=filename, local_dir=target_dir
                )
            except Exception as e:
                print(f"Warning: Failed to download {filename}: {e}")

    # Download voice style if missing
    voice_style_path_list = [os.path.join(target_dir, "voice_styles", f"{spk_id}.json") for spk_id in ["M1", "M2", "F1", "F2"]]
    for voice_style_path in voice_style_path_list:
        if not os.path.exists(voice_style_path):
            print(f"Downloading {voice_style_path}...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID, filename="/".join(voice_style_path.split("/")[-2:]), local_dir=target_dir
                )
            except Exception as e:
                print(f"Warning: Failed to download voice style: {e}")


def load_text_to_speech(
    model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16", use_gpu: bool = False
) -> TextToSpeech:
    # Load MNN settings (e.g., backend, thread num, precision, memory type) from config.json
    mnn_cfg_path = os.path.join(model_dir, "config.json")
    mnn_cfg = dict()
    mnn_backend_mapping = {
        'cpu': 0,
        'metal': 1,
        'cuda': 2,
        'opencl': 3,
        'opengl': 6,
        'vulkan': 7,
        'hiai': 8,
        'trt': 9,
    }
    with open(mnn_cfg_path, "r") as f:
        data = json.load(f)
        mnn_cfg["backend"] = mnn_backend_mapping[data['mnn_cfg_backend']]
        mnn_cfg["thread_num"] = data['mnn_cfg_thread_num']
        mnn_cfg["precision"] = data['mnn_cfg_precision']
        mnn_cfg["memory"] = data['mnn_cfg_memory']

    # New structure: mnn_models/{precision}/*.mnn
    models_dir = os.path.join(model_dir, "mnn_models")
    precision_dir = os.path.join(models_dir, precision)

    # Load Config
    cfg_path = os.path.join(models_dir, "tts.json")
    with open(cfg_path, "r") as f:
        cfgs = json.load(f)

    # Load Models from precision directory
    dp_path = os.path.join(precision_dir, "duration_predictor.mnn")
    text_enc_path = os.path.join(precision_dir, "text_encoder.mnn")
    vector_est_path = os.path.join(precision_dir, "vector_estimator.mnn")
    vocoder_path = os.path.join(precision_dir, "vocoder.mnn")

    # Define input/output names
    dp_ort = load_mnn(dp_path, ["text_ids", "style_dp", "text_mask"], ["duration"], mnn_cfg)
    text_enc_ort = load_mnn(
        text_enc_path, ["text_ids", "style_ttl", "text_mask"], ["text_emb"], mnn_cfg
    )

    # Note: vector_est_ort input names must match the order expected by the model
    # and enforced by our MNNInference class in engine.py
    vector_est_ort = load_mnn(
        vector_est_path,
        [
            "noisy_latent",
            "text_emb",
            "style_ttl",
            "latent_mask",
            "text_mask",
            "current_step",
            "total_step",
        ],
        ["denoised_latent"],
        mnn_cfg,
    )

    vocoder_ort = load_mnn(vocoder_path, ["latent"], ["wav_tts"], mnn_cfg)

    # Load Text Processor
    unicode_indexer_path = os.path.join(models_dir, "unicode_indexer.json")
    text_processor = UnicodeProcessor(unicode_indexer_path)

    return TextToSpeech(
        cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
    )


def get_voice_style_path(voice_name: str, model_dir: str = DEFAULT_CACHE_DIR) -> str:
    # Check if voice_name is a path
    if os.path.exists(voice_name):
        return voice_name

    # Check if it's a name in voice_styles directory
    style_path = os.path.join(model_dir, "voice_styles", f"{voice_name}.json")
    if os.path.exists(style_path):
        return style_path

    # Try case-insensitive
    styles_dir = os.path.join(model_dir, "voice_styles")
    if os.path.exists(styles_dir):
        for f in os.listdir(styles_dir):
            if f.lower() == f"{voice_name.lower()}.json":
                return os.path.join(styles_dir, f)

    raise ValueError(f"Voice style '{voice_name}' not found in {styles_dir}")
