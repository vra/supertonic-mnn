#!/usr/bin/env python3
"""Convert Supertonic ONNX models to MNN format for v2 and v3."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


HF_REPOS = {
    "v2": "Supertone/supertonic-2",
    "v3": "Supertone/supertonic-3",
}

MODEL_FILES = [
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
]

CONFIG_FILES = [
    "tts.json",
    "unicode_indexer.json",
]

VOICE_STYLES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

MNNCONVERT = "/opt/homebrew/opt/python@3.10/bin/python3.10"


def run_mnnconvert(onnx_path: str, mnn_path: str, fp16: bool = False, weight_quant_bits: int = 0):
    args = f"sys.argv = ['MNNConvert', '-f', 'ONNX', '--modelFile', '{onnx_path}', '--MNNModel', '{mnn_path}'"
    if fp16:
        args += ", '--fp16'"
    if weight_quant_bits > 0:
        args += f", '--weightQuantBits', '{weight_quant_bits}'"
    args += "]; main()"
    cmd = [
        MNNCONVERT, "-c",
        f"from MNN.tools.mnnconvert import main; import sys; {args}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    return True


def convert_version(version: str, output_dir: str):
    repo_id = HF_REPOS[version]
    version_dir = os.path.join(output_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Converting {version} from {repo_id}")
    print(f"{'='*60}")

    # Download ONNX models
    print("\n[1/4] Downloading ONNX models...")
    onnx_dir = os.path.join(version_dir, "onnx_tmp")
    os.makedirs(onnx_dir, exist_ok=True)
    for model_file in MODEL_FILES:
        print(f"  Downloading {model_file}...")
        path = hf_hub_download(repo_id, f"onnx/{model_file}")
        shutil.copy2(path, os.path.join(onnx_dir, model_file))

    # Download config files
    print("\n[2/4] Downloading config files...")
    models_dir = os.path.join(version_dir, "mnn_models")
    os.makedirs(models_dir, exist_ok=True)
    for cfg_file in CONFIG_FILES:
        path = hf_hub_download(repo_id, f"onnx/{cfg_file}")
        shutil.copy2(path, os.path.join(models_dir, cfg_file))
        print(f"  Copied {cfg_file}")

    # Download voice styles
    print("\n[3/4] Downloading voice styles...")
    styles_dir = os.path.join(version_dir, "voice_styles")
    os.makedirs(styles_dir, exist_ok=True)
    for style in VOICE_STYLES:
        try:
            path = hf_hub_download(repo_id, f"voice_styles/{style}.json")
            shutil.copy2(path, os.path.join(styles_dir, f"{style}.json"))
            print(f"  Downloaded {style}.json")
        except Exception:
            pass  # Some styles may not exist for all versions

    # Convert to MNN (fp32, fp16, int8)
    print("\n[4/4] Converting ONNX to MNN...")
    precisions = {"fp32": False, "fp16": True}

    for precision, use_fp16 in precisions.items():
        precision_dir = os.path.join(models_dir, precision)
        os.makedirs(precision_dir, exist_ok=True)
        print(f"\n  --- {precision} ---")
        for model_file in MODEL_FILES:
            model_name = model_file.replace(".onnx", "")
            onnx_path = os.path.join(onnx_dir, model_file)
            mnn_path = os.path.join(precision_dir, f"{model_name}.mnn")
            print(f"  Converting {model_name} -> {precision}...", end=" ")
            if run_mnnconvert(onnx_path, mnn_path, fp16=use_fp16):
                size_mb = os.path.getsize(mnn_path) / (1024 * 1024)
                print(f"OK ({size_mb:.1f} MB)")
            else:
                print("FAILED")

    # int8 quantization: use --weightQuantBits 8
    int8_dir = os.path.join(models_dir, "int8")
    os.makedirs(int8_dir, exist_ok=True)
    print(f"\n  --- int8 (weightQuantBits=8) ---")
    for model_file in MODEL_FILES:
        model_name = model_file.replace(".onnx", "")
        onnx_path = os.path.join(onnx_dir, model_file)
        mnn_path = os.path.join(int8_dir, f"{model_name}.mnn")
        print(f"  Converting {model_name} -> int8...", end=" ")
        if run_mnnconvert(onnx_path, mnn_path, weight_quant_bits=8):
            size_mb = os.path.getsize(mnn_path) / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
        else:
            print("FAILED")

    # Cleanup temp ONNX files
    shutil.rmtree(onnx_dir)
    print(f"\n  Cleaned up temp ONNX files")
    print(f"  Output: {version_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert Supertonic ONNX models to MNN")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="converted_models",
        help="Output directory (default: converted_models)",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["v2", "v3"],
        default=["v2", "v3"],
        help="Versions to convert (default: v2 v3)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Versions to convert: {args.versions}")

    for version in args.versions:
        convert_version(version, output_dir)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload yunfengwang/supertonic-tts-mnn {output_dir} .")


if __name__ == "__main__":
    main()
