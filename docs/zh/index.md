# 介绍

**Supertonic MNN** 是一个基于 [MNN](https://github.com/alibaba/MNN) 的高性能、轻量级文本转语音 (TTS) 推理库。它专为在移动端和嵌入式设备等各种平台上高效运行而设计，采用了 **Supertonic** TTS 模型架构。

## 主要特性

*   **极速推理**: 利用 MNN 的优化技术，提供低延迟的语音生成能力。
*   **极致轻量**: 依赖极少，易于部署。
*   **跨平台**: 支持 Linux, macOS, 和 Windows。
*   **Python API**: 提供易用的 Python 接口，方便无缝集成。
*   **CLI 支持**: 内置命令行工具，便于快速测试和批量处理。

## 架构

本库由多个协同工作的 MNN 模型组成：

*   **时长预测器 (Duration Predictor)**: 确定音素时长。
*   **文本编码器 (Text Encoder)**: 将输入文本编码为嵌入向量。
*   **向量估计器 (Vector Estimator)**: 预测声学特征 (Latent)。
*   **声码器 (Vocoder)**: 将特征转换为音频波形。

## 快速开始

请查看 [安装](installation.md) 指南以安装本库，或直接前往 [使用指南](usage.md) 章节开始生成语音。

## 致谢

本项目基于 Supertone Inc. 的原始 [Supertonic](https://github.com/supertone-inc/supertonic/) 工作。
