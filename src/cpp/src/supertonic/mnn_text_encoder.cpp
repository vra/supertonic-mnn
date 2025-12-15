#include "supertonic/mnn_text_encoder.hpp"
#include "mnn_tts_logger.hpp"

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cstring>

using namespace MNN::Express;

MNNTextEncoder::MNNTextEncoder(const std::string& model_path) {
    // 创建执行器
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    
    executor_ = std::shared_ptr<MNN::Express::Executor>(
        MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 4)
    );
    
    MNN::Express::ExecutorScope scope(executor_);
    
    // 加载模型
    module_ = std::shared_ptr<MNN::Express::Module>(
        MNN::Express::Module::load(input_names_, output_names_, model_path.c_str())
    );
    
    if (!module_) {
        PLOG(ERROR, "Failed to load Text Encoder MNN model: " + model_path);
        throw std::runtime_error("Failed to load Text Encoder MNN model: " + model_path);
    }
    
    PLOG(INFO, "Successfully loaded Text Encoder MNN model: " + model_path);
}

std::vector<float> MNNTextEncoder::encode(
    const std::vector<int>& text_ids,
    const std::vector<std::vector<float>>& style_ttl,
    const std::vector<float>& text_mask) {
    
    MNN::Express::ExecutorScope scope(executor_);
    
    // 准备输入
    std::vector<MNN::Express::VARP> inputs(3);
    
    // 输入文本IDs
    std::vector<int> text_ids_shape = {1, static_cast<int>(text_ids.size())};
    inputs[0] = _Const(text_ids.data(), text_ids_shape, NCHW, halide_type_of<int>());
    
    // 输入风格向量
    std::cout << "ttl size0: " << style_ttl.size() << std::endl;
    std::cout << "ttl size1: " << style_ttl[0].size() << std::endl;
    std::vector<int> style_ttl_shape = {1, static_cast<int>(style_ttl.size()), static_cast<int>(style_ttl[0].size())};
    inputs[1] = _Const(style_ttl.data(), style_ttl_shape, NCHW, halide_type_of<float>());
    
    // 输入文本掩码
    std::vector<int> text_mask_shape = {1, 1, static_cast<int>(text_mask.size())};
    inputs[2] = _Const(text_mask.data(), text_mask_shape, NCHW, halide_type_of<float>());
    
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
    
    std::cout <<"text encoder: " << std::endl;

    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i]<< std::endl;
        if (i > 10) {
            break;
        }
    }
    
    return result;
}