# Introduction

**Supertonic MNN** is a high-performance, lightweight text-to-speech (TTS) inference library based on [MNN](https://github.com/alibaba/MNN). It is designed to run efficiently on various platforms, including mobile and embedded devices, using the **Supertonic** TTS model architecture.

## Key Features

*   **Fast Inference**: Leveraging MNN's optimization, it delivers low-latency speech generation.
*   **Lightweight**: Minimal dependencies, easy to deploy.
*   **Cross-Platform**: Works on Linux, macOS, and Windows.
*   **Python API**: Easy-to-use Python interface for valid seamless integration.
*   **CLI Support**: Built-in command-line tool for quick testing and batch processing.

## Architecture

The library consists of several MNN models working together:

*   **Duration Predictor**: Determines phoneme durations.
*   **Text Encoder**: Encodes input text into embeddings.
*   **Vector Estimator**: Predicts acoustic features (Latent).
*   **Vocoder**: Converts features into audio waveforms.

## Getting Started

Check out the [Installation](installation.md) guide to set up the library, or go directly to the [Usage](usage.md) section to start generating speech.

## Acknowledgments

This project is based on the original [Supertonic](https://github.com/supertone-inc/supertonic/) by Supertone Inc.
