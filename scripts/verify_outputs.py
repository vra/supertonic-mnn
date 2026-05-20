#!/usr/bin/env python3
"""Verify MNN model outputs match ONNX outputs for supertonic v2/v3."""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


HF_REPOS = {
    "v2": "Supertone/supertonic-2",
    "v3": "Supertone/supertonic-3",
}

MODEL_NAMES = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]


def load_onnx_session(repo_id: str, model_name: str) -> ort.InferenceSession:
    path = hf_hub_download(repo_id, f"onnx/{model_name}.onnx")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def load_mnn_session(mnn_dir: str, model_name: str, precision: str = "fp32"):
    import MNN

    model_path = os.path.join(mnn_dir, "mnn_models", precision, f"{model_name}.mnn")
    config = {
        "backend": 0,  # CPU
        "thread_num": 4,
        "precision": "normal",
        "memory": "normal",
    }
    rt = MNN.nn.create_runtime_manager((config,))
    sess = ort.InferenceSession.__new__(ort.InferenceSession)
    # Use our MNNInference wrapper instead
    from supertonic_mnn.engine import MNNInference
    return model_path, config


def prepare_test_inputs(version: str, repo_id: str):
    """Prepare deterministic test inputs matching the model signatures."""
    np.random.seed(42)

    # Load unicode indexer
    path = hf_hub_download(repo_id, "onnx/unicode_indexer.json")
    with open(path) as f:
        indexer = json.load(f)

    # Prepare a simple text input with language tag
    lang = "en"
    text = f"<{lang}>Hello world.</{lang}>"
    text_ids = np.array([[indexer[ord(c)] for c in text]], dtype=np.int64)
    text_mask = np.ones((1, 1, len(text)), dtype=np.float32)

    # Load a voice style
    style_path = hf_hub_download(repo_id, "voice_styles/M1.json")
    with open(style_path) as f:
        style_data = json.load(f)

    style_ttl = np.array(style_data["style_ttl"]["data"], dtype=np.float32).reshape(
        1, *style_data["style_ttl"]["dims"][1:]
    )
    style_dp = np.array(style_data["style_dp"]["data"], dtype=np.float32).reshape(
        1, *style_data["style_dp"]["dims"][1:]
    )

    return {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "style_ttl": style_ttl,
        "style_dp": style_dp,
    }


def verify_model(
    model_name: str,
    onnx_session: ort.InferenceSession,
    mnn_model_path: str,
    mnn_config: dict,
    inputs: dict,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    """Compare ONNX and MNN outputs for a single model."""
    from supertonic_mnn.engine import MNNInference

    # Get input/output names from ONNX session
    input_names = [inp.name for inp in onnx_session.get_inputs()]
    output_names = [out.name for out in onnx_session.get_outputs()]

    # Prepare input dict for this model
    model_inputs = {name: inputs[name] for name in input_names if name in inputs}

    # Run ONNX
    onnx_outputs = onnx_session.run(None, model_inputs)

    # Run MNN
    mnn_session = MNNInference(mnn_model_path, input_names, output_names, mnn_config)
    mnn_outputs = mnn_session.run(output_names, model_inputs)

    # Compare
    for i, (onnx_out, mnn_out) in enumerate(zip(onnx_outputs, mnn_outputs)):
        onnx_out = np.array(onnx_out)
        mnn_out = np.array(mnn_out)

        if onnx_out.shape != mnn_out.shape:
            # MNN may flatten or reshape differently
            mnn_out = mnn_out.reshape(onnx_out.shape)

        max_diff = np.max(np.abs(onnx_out - mnn_out))
        mean_diff = np.mean(np.abs(onnx_out - mnn_out))
        close = np.allclose(onnx_out, mnn_out, atol=atol, rtol=rtol)

        return {
            "output_name": output_names[i],
            "shape": onnx_out.shape,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "close": close,
            "onnx_range": (float(onnx_out.min()), float(onnx_out.max())),
            "mnn_range": (float(mnn_out.min()), float(mnn_out.max())),
        }


def verify_duration_predictor(version: str, mnn_dir: str, precision: str = "fp32"):
    """Verify duration_predictor model."""
    from supertonic_mnn.engine import MNNInference

    repo_id = HF_REPOS[version]
    inputs = prepare_test_inputs(version, repo_id)

    # ONNX
    onnx_sess = load_onnx_session(repo_id, "duration_predictor")
    onnx_inputs = {
        "text_ids": inputs["text_ids"],
        "style_dp": inputs["style_dp"],
        "text_mask": inputs["text_mask"],
    }
    onnx_out = onnx_sess.run(None, onnx_inputs)

    # MNN
    model_path = os.path.join(mnn_dir, "mnn_models", precision, "duration_predictor.mnn")
    config = {"backend": 0, "thread_num": 4, "precision": "normal", "memory": "normal"}
    input_names = ["text_ids", "style_dp", "text_mask"]
    output_names = ["duration"]
    mnn_sess = MNNInference(model_path, input_names, output_names, config)
    mnn_out = mnn_sess.run(output_names, onnx_inputs)

    onnx_val = np.array(onnx_out[0]).flatten()
    mnn_val = np.array(mnn_out[0]).flatten()

    max_diff = np.max(np.abs(onnx_val - mnn_val))
    print(f"  duration_predictor: ONNX={onnx_val}, MNN={mnn_val}, diff={max_diff:.6f}")
    return max_diff


def verify_text_encoder(version: str, mnn_dir: str, precision: str = "fp32"):
    """Verify text_encoder model."""
    from supertonic_mnn.engine import MNNInference

    repo_id = HF_REPOS[version]
    inputs = prepare_test_inputs(version, repo_id)

    # ONNX
    onnx_sess = load_onnx_session(repo_id, "text_encoder")
    onnx_inputs = {
        "text_ids": inputs["text_ids"],
        "style_ttl": inputs["style_ttl"],
        "text_mask": inputs["text_mask"],
    }
    onnx_out = onnx_sess.run(None, onnx_inputs)

    # MNN
    model_path = os.path.join(mnn_dir, "mnn_models", precision, "text_encoder.mnn")
    config = {"backend": 0, "thread_num": 4, "precision": "normal", "memory": "normal"}
    input_names = ["text_ids", "style_ttl", "text_mask"]
    output_names = ["text_emb"]
    mnn_sess = MNNInference(model_path, input_names, output_names, config)
    mnn_out = mnn_sess.run(output_names, onnx_inputs)

    onnx_val = np.array(onnx_out[0])
    mnn_val = np.array(mnn_out[0]).reshape(onnx_val.shape)

    max_diff = np.max(np.abs(onnx_val - mnn_val))
    mean_diff = np.mean(np.abs(onnx_val - mnn_val))
    print(f"  text_encoder: shape={onnx_val.shape}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    return max_diff


def verify_vocoder(version: str, mnn_dir: str, precision: str = "fp32"):
    """Verify vocoder model with a small random latent input."""
    from supertonic_mnn.engine import MNNInference

    repo_id = HF_REPOS[version]
    np.random.seed(42)

    latent = np.random.randn(1, 144, 10).astype(np.float32)

    # ONNX
    onnx_sess = load_onnx_session(repo_id, "vocoder")
    onnx_out = onnx_sess.run(None, {"latent": latent})

    # MNN
    model_path = os.path.join(mnn_dir, "mnn_models", precision, "vocoder.mnn")
    config = {"backend": 0, "thread_num": 4, "precision": "normal", "memory": "normal"}
    mnn_sess = MNNInference(model_path, ["latent"], ["wav_tts"], config)
    mnn_out = mnn_sess.run(["wav_tts"], {"latent": latent})

    onnx_val = np.array(onnx_out[0])
    mnn_val = np.array(mnn_out[0]).reshape(onnx_val.shape)

    max_diff = np.max(np.abs(onnx_val - mnn_val))
    mean_diff = np.mean(np.abs(onnx_val - mnn_val))
    print(f"  vocoder: shape={onnx_val.shape}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    return max_diff


def main():
    parser = argparse.ArgumentParser(description="Verify MNN models match ONNX outputs")
    parser.add_argument(
        "--mnn-dir",
        type=str,
        default="converted_models",
        help="Directory containing converted MNN models",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["v2", "v3"],
        default=["v2", "v3"],
        help="Versions to verify",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp32",
        help="MNN precision to verify (default: fp32)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Supertonic MNN vs ONNX Verification")
    print("=" * 60)

    all_passed = True
    for version in args.versions:
        mnn_dir = os.path.join(args.mnn_dir, version)
        if not os.path.exists(mnn_dir):
            print(f"\n[SKIP] {version}: {mnn_dir} not found")
            continue

        print(f"\n--- {version} (precision={args.precision}) ---")

        try:
            diff = verify_duration_predictor(version, mnn_dir, args.precision)
            if diff > 0.01:
                print(f"    WARNING: large diff!")
                all_passed = False
        except Exception as e:
            print(f"  duration_predictor: FAILED - {e}")
            all_passed = False

        try:
            diff = verify_text_encoder(version, mnn_dir, args.precision)
            if diff > 0.01:
                print(f"    WARNING: large diff!")
                all_passed = False
        except Exception as e:
            print(f"  text_encoder: FAILED - {e}")
            all_passed = False

        try:
            diff = verify_vocoder(version, mnn_dir, args.precision)
            if diff > 0.05:
                print(f"    WARNING: large diff!")
                all_passed = False
        except Exception as e:
            print(f"  vocoder: FAILED - {e}")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS HAD WARNINGS - review output above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
