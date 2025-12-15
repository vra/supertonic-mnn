#pragma once

/**
 * @file mnn_text_encoder.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief Supertonic TTS Text Encoder模型推理类
 */

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <string>
#include <vector>
#include <memory>

using namespace MNN::Express;

class MNNTextEncoder {
public:
    MNNTextEncoder(const std::string& model_path);
    
    std::vector<float> encode(
        const std::vector<int>& text_ids,
        const std::vector<std::vector<float>>& style_ttl,
        const std::vector<float>& text_mask);

private:
    std::shared_ptr<Module> module_;
    std::shared_ptr<Executor> executor_;
    
    // 输入输出名称
    const std::vector<std::string> input_names_{"text_ids", "style_ttl", "text_mask"};
    const std::vector<std::string> output_names_{"text_emb"};
};