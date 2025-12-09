# 安装指南

## 前置要求

*   Python 3.9 或更高版本
*   pip

## 从 PyPI 安装

目前，`supertonic-mnn` 可以通过 PyPI 安装 (示例，如果尚未发布，请调整或提供 git 安装方式):

```bash
pip install supertonic-mnn
```

## 从源码安装

要从 GitHub 安装最新版本：

```bash
git clone https://github.com/vra/supertonic-mnn.git
cd supertonic-mnn
pip install .
```

## 依赖项

安装过程中会自动安装以下依赖项：

*   `mnn>=2.0.0`
*   `numpy`
*   `soundfile`
*   `huggingface_hub`
*   `tqdm`
*   `requests`
