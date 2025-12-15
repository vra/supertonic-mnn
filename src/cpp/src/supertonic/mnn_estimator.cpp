#include "supertonic/mnn_estimator.hpp"
#include "mnn_tts_logger.hpp"

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cstring>

using namespace MNN::Express;

MNNVectorEstimator::MNNVectorEstimator(const std::string &model_path) {
  // 创建执行器
  MNN::BackendConfig backendConfig;
  backendConfig.precision = MNN::BackendConfig::Precision_Normal;
  backendConfig.memory = MNN::BackendConfig::Memory_Normal;

  executor_ = std::shared_ptr<MNN::Express::Executor>(
      MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 4));

  MNN::Express::ExecutorScope scope(executor_);

  // 加载模型
  module_ = std::shared_ptr<MNN::Express::Module>(MNN::Express::Module::load(
      input_names_, output_names_, model_path.c_str()));

  if (!module_) {
    PLOG(ERROR, "Failed to load Vector Estimator MNN model: " + model_path);
    throw std::runtime_error("Failed to load Vector Estimator MNN model: " +
                             model_path);
  }

  PLOG(INFO, "Successfully loaded Vector Estimator MNN model: " + model_path);
}

std::vector<float> MNNVectorEstimator::estimate(
    const std::vector<float> &noisy_latent, const std::vector<float> &text_emb,
    const std::vector<float> &style_ttl, const std::vector<int> &latent_mask,
    const std::vector<float> &text_mask, int current_step, int total_step) {

  MNN::Express::ExecutorScope scope(executor_);

  // 准备输入
  std::vector<MNN::Express::VARP> inputs(7);

  /*
  [ noisy_latent ]: dimensionFormat: NCHW, size: [ 1,144,-1 ], type is float
  [ text_emb ]: dimensionFormat: NCHW, size: [ 1,256,-1 ], type is float
  [ style_ttl ]: dimensionFormat: NCHW, size: [ 1,50,256 ], type is float
  [ latent_mask ]: dimensionFormat: NCHW, size: [ 1,1,-1 ], type is float
  [ text_mask ]: dimensionFormat: NCHW, size: [ 1,1,-1 ], type is float
  [ current_step ]: dimensionFormat: NCHW, size: [ 1 ], type is float
  [ total_step ]: dimensionFormat: NCHW, size: [ 1 ], type is float
  */

  // 输入噪声潜变量
  int latent_dim = 144;
  if (noisy_latent.size() % latent_dim != 0) {
    PLOG(ERROR, "Noisy latent size mismatch");
    return {};
  }
  int latent_len = noisy_latent.size() / latent_dim;
  std::vector<int> noisy_latent_shape = {1, latent_dim, latent_len};
  inputs[0] = _Const(noisy_latent.data(), noisy_latent_shape, NCHW,
                     halide_type_of<float>());

  // 输入文本嵌入
  int text_dim = 256;
  if (text_emb.size() % text_dim != 0) {
    PLOG(ERROR, "Text emb size mismatch");
    return {};
  }
  int text_len = text_emb.size() / text_dim;
  std::vector<int> text_emb_shape = {1, text_dim, text_len};
  inputs[1] =
      _Const(text_emb.data(), text_emb_shape, NCHW, halide_type_of<float>());

  // 输入风格向量
  std::vector<int> style_ttl_shape = {1, 50, 256};
  inputs[2] =
      _Const(style_ttl.data(), style_ttl_shape, NCHW, halide_type_of<float>());

  // 输入潜变量掩码
  std::vector<int> latent_mask_shape = {1, 1,
                                        static_cast<int>(latent_mask.size())};
  inputs[3] =
      _Const(latent_mask.data(), latent_mask_shape, NCHW,
             halide_type_of<int>()); // Note: mask is usually int or float?
                                     // Python says float32 for latent_mask.
  // C++ signature says std::vector<int> latent_mask.
  // If model expects float, we need to cast or Change signature.
  // Python: latent_mask = length_to_mask(...) -> float32.
  // My C++ TextProcessor code made text_mask float.
  // But MNNVectorEstimator::estimate takes vector<int> latent_mask.
  // I should check signature in mnn_estimator.hpp.
  // If implementation takes vector<int>, I pass int.
  // BUT MNN `inputs[3] = ... halide_type_of<int>()` creates int tensor.
  // If model expects float mask, this will fail or cast.
  // Let's check Python engine.py: `latent_mask: (B, 1, max_len)` float32.
  // So I MUST pass float mask.
  // I need to change signature of `estimate` to take float mask or cast it
  // inside. Since I can't easily change all signatures without touching headers
  // and callers, I will cast it here if valid. But wait, `latent_mask` passed
  // from `mnn_supertonic_tts_impl.cpp` was `std::vector<int>`. I should change
  // `mnn_supertonic_tts_impl.cpp` to use `std::vector<float>` for latent mask
  // if possible, or cast here. Casting here:

  std::vector<float> latent_mask_float(latent_mask.begin(), latent_mask.end());
  inputs[3] = _Const(latent_mask_float.data(), latent_mask_shape, NCHW,
                     halide_type_of<float>());

  // 输入文本掩码
  std::vector<int> text_mask_shape = {1, 1, static_cast<int>(text_mask.size())};
  inputs[4] =
      _Const(text_mask.data(), text_mask_shape, NCHW, halide_type_of<float>());

  // 当前步数
  float current_step_val = static_cast<float>(current_step);
  inputs[5] = _Const(&current_step_val, {1}, NCHW, halide_type_of<float>());

  // 总步数
  float total_step_val = static_cast<float>(total_step);
  inputs[6] = _Const(&total_step_val, {1}, NCHW, halide_type_of<float>());

  // 执行推理
  auto outputs = module_->onForward(inputs);

  // 处理输出
  if (outputs.empty()) {
    PLOG(ERROR, "Vector Estimator returned empty output");
    return {};
  }

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));

  return result;
}