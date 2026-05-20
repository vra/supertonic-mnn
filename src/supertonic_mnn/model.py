import os
import json
import time
from huggingface_hub import hf_hub_download
from .engine import TextToSpeech, load_mnn
from .text import UnicodeProcessor

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/supertonic-mnn")
REPO_ID = "yunfengwang/supertonic-tts-mnn"

VOICE_STYLES_ALL = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]


def _version_prefix(version: str) -> str:
    if version in ("v2", "v3"):
        return f"{version}/"
    return ""


def ensure_models(target_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16", version: str = "v3"):
    """
    Ensure that the MNN models and voice styles are present in the target directory.
    If not, download them from Hugging Face.

    Args:
        target_dir: Directory to store models.
        precision: Model precision ('fp16', 'fp32', 'int8').
        version: Model version ('v1', 'v2', 'v3'). Default: 'v3'.
    """
    print(f"Checking models in {target_dir} (version={version}, precision={precision})...")

    prefix = _version_prefix(version)
    version_dir = os.path.join(target_dir, version) if version in ("v2", "v3") else target_dir
    models_dir = os.path.join(version_dir, "mnn_models")
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
        print(f"All required models found in {version_dir}")
        return

    print(f"Missing files: {missing_files}")
    print(f"Attempting to download from HF ({REPO_ID}) version={version}, precision={precision}...")

    # Download config.json (shared across versions)
    try:
        if not os.path.exists(os.path.join(target_dir, "config.json")):
            print("Downloading config.json...")
            hf_hub_download(
                repo_id=REPO_ID, filename="config.json", local_dir=target_dir
            )
    except Exception as e:
        print(f"Warning: Failed to download config.json: {e}")

    # Download model config files
    try:
        if not os.path.exists(os.path.join(models_dir, "tts.json")):
            print("Downloading tts.json...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{prefix}mnn_models/tts.json",
                local_dir=target_dir,
            )

        if not os.path.exists(os.path.join(models_dir, "unicode_indexer.json")):
            print("Downloading unicode_indexer.json...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{prefix}mnn_models/unicode_indexer.json",
                local_dir=target_dir,
            )
    except Exception as e:
        print(f"Failed to download config files: {e}")
        if missing_files:
            raise RuntimeError(
                f"Models missing in {version_dir} and download failed."
            ) from e

    # Download precision-specific model files
    model_files = [
        f"{prefix}mnn_models/{precision}/duration_predictor.mnn",
        f"{prefix}mnn_models/{precision}/text_encoder.mnn",
        f"{prefix}mnn_models/{precision}/vector_estimator.mnn",
        f"{prefix}mnn_models/{precision}/vocoder.mnn",
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

    # Download voice styles
    voice_styles_dir = os.path.join(version_dir, "voice_styles")
    for spk_id in VOICE_STYLES_ALL:
        voice_style_path = os.path.join(voice_styles_dir, f"{spk_id}.json")
        if not os.path.exists(voice_style_path):
            hf_filename = f"{prefix}voice_styles/{spk_id}.json"
            try:
                hf_hub_download(
                    repo_id=REPO_ID, filename=hf_filename, local_dir=target_dir
                )
            except Exception:
                pass  # Some styles may not exist for all versions


def load_text_to_speech(
    model_dir: str = DEFAULT_CACHE_DIR, precision: str = "fp16", use_gpu: bool = False, version: str = "v3"
) -> TextToSpeech:
    # Load MNN settings from config.json
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

    # Versioned model directory
    version_dir = os.path.join(model_dir, version) if version in ("v2", "v3") else model_dir
    models_dir = os.path.join(version_dir, "mnn_models")
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


def get_voice_style_path(voice_name: str, model_dir: str = DEFAULT_CACHE_DIR, version: str = "v3") -> str:
    # Check if voice_name is a path
    if os.path.exists(voice_name):
        return voice_name

    # Check in versioned directory first
    version_dir = os.path.join(model_dir, version) if version in ("v2", "v3") else model_dir
    style_path = os.path.join(version_dir, "voice_styles", f"{voice_name}.json")
    if os.path.exists(style_path):
        return style_path

    # Fallback to root voice_styles directory
    style_path = os.path.join(model_dir, "voice_styles", f"{voice_name}.json")
    if os.path.exists(style_path):
        return style_path

    # Try case-insensitive in versioned directory
    styles_dir = os.path.join(version_dir, "voice_styles")
    if os.path.exists(styles_dir):
        for f in os.listdir(styles_dir):
            if f.lower() == f"{voice_name.lower()}.json":
                return os.path.join(styles_dir, f)

    raise ValueError(f"Voice style '{voice_name}' not found in {styles_dir}")
