#pragma once

/**
 * @file mnn_estimator.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief Supertonic TTS Vector Estimator模型推理类
 */

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <string>
#include <vector>
#include <memory>

using namespace MNN::Express;

class MNNVectorEstimator {
public:
    MNNVectorEstimator(const std::string& model_path);
    
    std::vector<float> estimate(
        const std::vector<float>& noisy_latent,
        const std::vector<float>& text_emb,
        const std::vector<float>& style_ttl,
        const std::vector<int>& latent_mask,
        const std::vector<float>& text_mask,
        int current_step,
        int total_step);

private:
    std::shared_ptr<Module> module_;
    std::shared_ptr<Executor> executor_;
    
    // 输入输出名称
    const std::vector<std::string> input_names_{
        "noisy_latent",
        "text_emb",
        "style_ttl",
        "latent_mask",
        "text_mask",
        "current_step",
        "total_step"
    };
    const std::vector<std::string> output_names_{"denoised_latent"};
};