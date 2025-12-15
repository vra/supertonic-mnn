#pragma once

/**
 * @file mnn_vocoder.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief Supertonic TTS Vocoder模型推理类
 */

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <string>
#include <vector>
#include <memory>

using namespace MNN::Express;

class MNNVocoder {
public:
    MNNVocoder(const std::string& model_path);
    
    std::vector<float> vocode(const std::vector<float>& latent);

private:
    std::shared_ptr<Module> module_;
    std::shared_ptr<Executor> executor_;
    
    // 输入输出名称
    const std::vector<std::string> input_names_{"latent"};
    const std::vector<std::string> output_names_{"wav_tts"};
};