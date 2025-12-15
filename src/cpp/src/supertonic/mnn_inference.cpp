/**
 * @file mnn_inference.cpp
 * @brief MNN推理器封装类实现
 */

#include "supertonic/mnn_inference.hpp"
#include "mnn_tts_logger.hpp"

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cstring>

using namespace MNN::Express;

MNNInference::MNNInference(const std::string& model_path, 
                           const std::vector<std::string>& input_names,
                           const std::vector<std::string>& output_names)
    : input_names_(input_names), output_names_(output_names) {
    
    // 创建执行器
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    
    executor_ = std::shared_ptr<MNN::Express::Executor>(
        MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 4)
    );
    
    MNN::Express::ExecutorScope scope(executor_);
    
    // 加载模型 - 修复API调用
    module_ = std::shared_ptr<MNN::Express::Module>(
        MNN::Express::Module::load(input_names_, output_names_, model_path.c_str())
    );
    
    if (!module_) {
        PLOG(ERROR, "Failed to load MNN model: " + model_path);
        throw std::runtime_error("Failed to load MNN model: " + model_path);
    }
    
    PLOG(INFO, "Successfully loaded MNN model: " + model_path);
}

MNNInference::~MNNInference() = default;

std::vector<std::vector<float>> MNNInference::run(const std::map<std::string, std::vector<float>>& input_dict) {
    MNN::Express::ExecutorScope scope(executor_);
    
    // 准备输入
    std::vector<MNN::Express::VARP> inputs(input_names_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
        auto it = input_dict.find(input_names_[i]);
        if (it == input_dict.end()) {
            throw std::runtime_error("Missing input: " + input_names_[i]);
        }
        
        const auto& data = it->second;
        // 使用正确的API创建常量张量
        std::vector<int> shape = {static_cast<int>(data.size())};
        inputs[i] = _Const(data.data(), shape, NHWC);
    }
    
    // 执行推理
    auto outputs = module_->onForward(inputs);
    
    // 处理输出
    std::vector<std::vector<float>> result;
    for (const auto& output : outputs) {
        auto size = output->getInfo()->size;
        std::vector<float> output_data(size);
        ::memcpy(output_data.data(), output->readMap<float>(), size * sizeof(float));
        result.push_back(std::move(output_data));
    }
    
    return result;
}

std::vector<std::vector<int>> MNNInference::runInt(const std::map<std::string, std::vector<int>>& input_dict) {
    MNN::Express::ExecutorScope scope(executor_);
    
    // 准备输入
    std::vector<MNN::Express::VARP> inputs(input_names_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
        auto it = input_dict.find(input_names_[i]);
        if (it == input_dict.end()) {
            throw std::runtime_error("Missing input: " + input_names_[i]);
        }
        
        const auto& data = it->second;
        // 使用正确的API创建常量张量
        std::vector<int> shape = {static_cast<int>(data.size())};
        inputs[i] = _Const(data.data(), shape, NHWC);
    }
    
    // 执行推理
    auto outputs = module_->onForward(inputs);
    
    // 处理输出
    std::vector<std::vector<int>> result;
    for (const auto& output : outputs) {
        auto size = output->getInfo()->size;
        std::vector<int> output_data(size);
        ::memcpy(output_data.data(), output->readMap<int>(), size * sizeof(int));
        result.push_back(std::move(output_data));
    }
    
    return result;
}