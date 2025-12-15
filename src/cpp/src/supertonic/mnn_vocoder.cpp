#include "supertonic/mnn_vocoder.hpp"
#include "mnn_tts_logger.hpp"

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cstring>

using namespace MNN::Express;

MNNVocoder::MNNVocoder(const std::string &model_path) {
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
    PLOG(ERROR, "Failed to load Vocoder MNN model: " + model_path);
    throw std::runtime_error("Failed to load Vocoder MNN model: " + model_path);
  }

  PLOG(INFO, "Successfully loaded Vocoder MNN model: " + model_path);
}

std::vector<float> MNNVocoder::vocode(const std::vector<float> &latent) {
  MNN::Express::ExecutorScope scope(executor_);

  // 准备输入
  std::vector<MNN::Express::VARP> inputs(1);

  // 输入潜变量
  // 输入潜变量
  int latent_dim = 144;
  if (latent.size() % latent_dim != 0) {
    PLOG(ERROR, "Latent size mismatch");
    return {};
  }
  int latent_len = latent.size() / latent_dim;
  std::vector<int> latent_shape = {1, latent_dim, latent_len};
  inputs[0] =
      _Const(latent.data(), latent_shape, NCHW, halide_type_of<float>());

  // 执行推理
  auto outputs = module_->onForward(inputs);

  // 处理输出
  if (outputs.empty()) {
    PLOG(ERROR, "Vocoder returned empty output");
    return {};
  }

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));

  return result;
}