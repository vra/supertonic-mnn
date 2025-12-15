#include "supertonic/mnn_text_encoder.hpp"
#include "mnn_tts_logger.hpp"

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cstring>

using namespace MNN::Express;

MNNTextEncoder::MNNTextEncoder(const std::string &model_path) {
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
    PLOG(ERROR, "Failed to load Text Encoder MNN model: " + model_path);
    throw std::runtime_error("Failed to load Text Encoder MNN model: " +
                             model_path);
  }

  PLOG(INFO, "Successfully loaded Text Encoder MNN model: " + model_path);
}

std::vector<float>
MNNTextEncoder::encode(const std::vector<int> &text_ids,
                       const std::vector<std::vector<float>> &style_ttl,
                       const std::vector<float> &text_mask) {

  MNN::Express::ExecutorScope scope(executor_);

  // 准备输入
  std::vector<MNN::Express::VARP> inputs(3);

  // 输入文本IDs
  int num_tokens = static_cast<int>(text_ids.size());
  std::vector<int> text_ids_shape = {1, num_tokens};
  inputs[0] = MNN::Express::_Input(text_ids_shape, NCHW, halide_type_of<int>());
  auto text_ids_ptr = inputs[0]->writeMap<int>();
  for (int i = 0; i < num_tokens; ++i) {
    text_ids_ptr[i] = text_ids[i];
  }

  // 输入风格向量
  int ttl_size_0 = static_cast<int>(style_ttl.size());
  int ttl_size_1 = static_cast<int>(style_ttl[0].size());
  std::vector<int> style_ttl_shape = {1, ttl_size_0, ttl_size_1};
  inputs[1] =
      MNN::Express::_Input(style_ttl_shape, NCHW, halide_type_of<float>());
  auto style_ttl_ptr = inputs[1]->writeMap<float>();
  for (int i = 0; i < ttl_size_0; ++i) {
    for (int j = 0; j < ttl_size_1; ++j) {
      style_ttl_ptr[i * ttl_size_1 + j] = style_ttl[i][j];
    }
  }

  // 输入文本掩码
  std::vector<int> text_mask_shape = {1, 1, static_cast<int>(text_mask.size())};
  inputs[2] =
      MNN::Express::_Input(text_mask_shape, NCHW, halide_type_of<float>());
  auto text_mask_ptr = inputs[2]->writeMap<float>();
  for (size_t i = 0; i < text_mask.size(); ++i) {
    text_mask_ptr[i] = text_mask[i];
  }

  // 执行推理
  std::cout << "begin text encoder " << std::endl;
  auto outputs = module_->onForward(inputs);
  std::cout << "end text encoder " << std::endl;

  // 处理输出
  if (outputs.empty()) {
    PLOG(ERROR, "Text Encoder returned empty output");
    return {};
  }

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));

  std::cout << "text encoder: " << std::endl;

  for (int i = 0; i < result.size(); i++) {
    std::cout << result[i] << std::endl;
    if (i > 10) {
      break;
    }
  }

  return result;
}