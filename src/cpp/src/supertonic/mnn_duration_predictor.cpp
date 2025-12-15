#include "supertonic/mnn_duration_predictor.hpp"
#include "mnn_tts_logger.hpp"

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cstring>

using namespace MNN::Express;

MNNDurationPredictor::MNNDurationPredictor(const std::string &model_path)
{
    // 创建执行器
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;

    executor_ = std::shared_ptr<MNN::Express::Executor>(
        MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 4));

    MNN::Express::ExecutorScope scope(executor_);

    // 加载模型
    module_ = std::shared_ptr<MNN::Express::Module>(
        MNN::Express::Module::load(input_names_, output_names_, model_path.c_str()));

    if (!module_)
    {
        PLOG(ERROR, "Failed to load Duration Predictor MNN model: " + model_path);
        throw std::runtime_error("Failed to load Duration Predictor MNN model: " + model_path);
    }

    PLOG(INFO, "Successfully loaded Duration Predictor MNN model: " + model_path);
}

std::vector<float> MNNDurationPredictor::predict(
    const std::vector<int> &text_ids,
    const std::vector<std::vector<float>> &style_dp,
    const std::vector<float> &text_mask)
{

    MNN::Express::ExecutorScope scope(executor_);

    // 准备输入
    std::vector<MNN::Express::VARP> inputs(3);
    int num_tokens = text_ids.size();
    int dp_size_0 = style_dp.size();
    int dp_size_1 = style_dp[0].size();
    std::cout <<"dp size1, 2:" << dp_size_0 << ", " << dp_size_1 << std::endl;

    // 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW
    inputs[0] = MNN::Express::_Input({1, num_tokens}, NCHW, halide_type_of<int>());
    inputs[1] = MNN::Express::_Input({1, dp_size_0, dp_size_1}, NCHW, halide_type_of<float>());
    inputs[2] = MNN::Express::_Input({1, 1, num_tokens}, NCHW, halide_type_of<float>());

    // 设置输入数据
    std::vector<int *> input_pointer0 = {inputs[0]->writeMap<int>(),
                                         };
    for (int i = 0; i < num_tokens; ++i)
    {
        input_pointer0[0][i] = text_ids[i];
        std::cout << text_ids[i] << std::endl;
    }

    std::vector<float *> input_pointer1 = {
        inputs[1]->writeMap<float>(),
        inputs[2]->writeMap<float>()
    };
    for (int i = 0; i < dp_size_0; ++i)
    {
        for (int j = 0; j < dp_size_1; j++)
        {
            input_pointer1[0][i*dp_size_1+j] = style_dp[i][j];
            std::cout << style_dp[i][j] << std::endl;
        }
    }
      for (int i = 0; i < num_tokens; ++i)
    {
        input_pointer1[1][i] = 1.0; 
    }

    // 执行推理
    std::cout << "begin forward" << std::endl;
    auto outputs = module_->onForward(inputs);
    std::cout << "end forward" << std::endl;

    // 处理输出
    if (outputs.empty())
    {
        PLOG(ERROR, "Duration Predictor returned empty output");
        return {};
    }

    auto output = outputs[0];
    auto size = output->getInfo()->size;
    std::vector<float> result(size);
    ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));
    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << std::endl;
    }

    return result;
}