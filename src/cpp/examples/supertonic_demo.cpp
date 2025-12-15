/**
 * @file supertonic_demo.cpp
 * @brief Supertonic TTS C++ SDK 使用示例
 */

#include "mnn_tts_logger.hpp"
#include "supertonic/mnn_supertonic_tts_impl.hpp"
#include <chrono>
#include <iostream>

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

int main() {
  try {
    // 设置日志级别
    MNNITTSLogger::GetInstance().SetLogLevel(INFO);

    // 初始化配置
    TTSConfig config;
    config.model_dir = "/Users/ws/supertonic-tts-mnn"; // 模型目录
    config.precision = "fp16";                         // 精度设置

    // 初始化TTS引擎
    auto t0 = clk::now();
    // MNNSupertonicTTSImpl(model_dir, precision, cache_dir)
    MNNSupertonicTTSImpl tts(config.model_dir, config.precision,
                             config.model_dir + "/cache");
    auto t1 = clk::now();

    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);
    PLOG(INFO, "TTS engine initialization time: " + std::to_string(d1.count()) +
                   " ms");

    // 测试文本
    std::string text = "It’s nice to have shortcuts for commands you use often. To have normalized names for installing dependencies, running the code, or starting the tests across all your projects.";
    std::string voice = "F1";

    // 文本转语音
    auto t2 = clk::now();
    auto [audio_data, sample_rate, rtf] =
        tts.synthesize(text, "F1", 5, 1.0, "en");
    auto t3 = clk::now();

    auto d2 = std::chrono::duration_cast<ms>(t3 - t2);
    PLOG(INFO, "Synthesis time: " + std::to_string(d2.count()) + " ms");
    PLOG(INFO,
         "Audio length: " + std::to_string(audio_data.size()) + " samples");
    PLOG(INFO, "Sample rate: " + std::to_string(sample_rate) + " Hz");
    PLOG(INFO, "Real Time Factor (RTF): " + std::to_string(rtf));

    // 保存音频文件
    std::string output_file = "output.wav";
    if (MNNSupertonicTTSImpl::save(output_file, audio_data, sample_rate)) {
      PLOG(INFO, "Audio saved to: " + output_file);
    } else {
      PLOG(ERROR, "Failed to save audio file!");
      return 1;
    }

    PLOG(INFO, "Demo completed successfully!");
    return 0;
  } catch (const std::exception &e) {
    PLOG(ERROR, "Error: " + std::string(e.what()));
    return 1;
  }
}