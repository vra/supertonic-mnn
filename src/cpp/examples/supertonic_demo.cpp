/**
 * @file supertonic_demo.cpp
 * @brief Supertonic TTS C++ SDK 使用示例
 */

#include "supertonic/supertonic.hpp"
#include "mnn_tts_logger.hpp"
#include <iostream>
#include <chrono>

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

int main() {
    try {
        // 设置日志级别
        MNNITTSLogger::GetInstance().SetLogLevel(INFO);
        
        // 初始化配置
        TTSConfig config;
        config.model_dir = "/Users/ws/supertonic-tts-mnn";  // 模型目录
        config.precision = "fp16";  // 精度设置
        
        // 初始化TTS引擎
        auto t0 = clk::now();
        SupertonicTTS tts(config);
        auto t1 = clk::now();
        
        auto d1 = std::chrono::duration_cast<ms>(t1 - t0);
        std::cout << "TTS engine initialization time: " << d1.count() << " ms" << std::endl;
        
        // 测试文本
        std::string text = "Nice to meet you";
        std::string voice = "default";
        
        // 文本转语音
        auto t2 = clk::now();
        auto [audio_data, sample_rate, rtf] = tts.synthesize(text, "M1", 5, 1.0, "en");
        auto t3 = clk::now();
        
        auto d2 = std::chrono::duration_cast<ms>(t3 - t2);
        std::cout << "Synthesis time: " << d2.count() << " ms" << std::endl;
        std::cout << "Audio length: " << audio_data.size() << " samples" << std::endl;
        std::cout << "Sample rate: " << sample_rate << " Hz" << std::endl;
        std::cout << "Real Time Factor (RTF): " << rtf << std::endl;
        
        // 保存音频文件
        std::string output_file = "output.wav";
        if (SupertonicTTS::save(output_file, audio_data, sample_rate)) {
            std::cout << "Audio saved to: " << output_file << std::endl;
        } else {
            std::cerr << "Failed to save audio file!" << std::endl;
            return 1;
        }
        
        std::cout << "Demo completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}