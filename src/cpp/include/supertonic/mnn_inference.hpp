/**
 * @file mnn_inference.hpp
 * @brief MNN推理器封装类
 */

#ifndef _HEADER_MNN_INFERENCE_HPP_
#define _HEADER_MNN_INFERENCE_HPP_

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <string>
#include <vector>
#include <memory>
#include <map>

using namespace MNN::Express;

class MNNInference {
public:
    MNNInference(const std::string& model_path, 
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names);
    
    ~MNNInference();
    
    std::vector<std::vector<float>> run(const std::map<std::string, std::vector<float>>& input_dict);
    std::vector<std::vector<int>> runInt(const std::map<std::string, std::vector<int>>& input_dict);

private:
    std::shared_ptr<MNN::Express::Module> module_;
    std::shared_ptr<MNN::Express::Executor> executor_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    // 辅助函数
    VARP createInputTensor(const std::vector<float>& data, const std::vector<int>& shape);
    VARP createInputTensorInt(const std::vector<int>& data, const std::vector<int>& shape);
};

#endif // _HEADER_MNN_INFERENCE_HPP_