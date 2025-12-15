/**
 * @file supertonic.hpp
 * @brief Supertonic TTS C++ SDK 主要头文件
 */

#ifndef _HEADER_SUPERTONIC_TTS_HPP_
#define _HEADER_SUPERTONIC_TTS_HPP_

#include <string>
#include <vector>
#include <memory>
#include <tuple>

#include "mnn_tts_impl_base.hpp"

// 音频数据类型定义
typedef std::vector<int16_t> Audio;

// 语音风格数据结构
struct VoiceStyle {
    std::vector<std::vector<float>> ttl;  // text-to-latent style vectors
    std::vector<std::vector<float>> dp;   // duration predictor style vectors
    
    VoiceStyle() = default;
    
    VoiceStyle(const std::vector<std::vector<float>>& ttl_data, 
               const std::vector<std::vector<float>>& dp_data) 
        : ttl(ttl_data), dp(dp_data) {}
};

// 配置参数结构体
struct TTSConfig {
    std::string model_dir;
    std::string precision;  // "fp32", "fp16", "int8"
    
    TTSConfig() : model_dir("/Users/ws/supertonic-tts-mnn"), precision("fp16") {}
    
    TTSConfig(const std::string& dir, const std::string& prec) 
        : model_dir(dir), precision(prec) {}
};

class SupertonicTTS {
public:
    // 构造函数
    explicit SupertonicTTS(const TTSConfig& config);
    
    // 析构函数
    ~SupertonicTTS();
    
    // 文本转语音核心接口
    std::tuple<std::vector<float>, int, float> synthesize(
        const std::string& text,
        const std::string& voice,
        int steps = 5,
        float speed = 1.0f,
        const std::string& language = "zh");
    
    // 保存音频文件
    static bool save(const std::string& filename, 
                     const std::vector<float>& audio_data, 
                     int sample_rate);

private:
    // Pimpl手法隐藏实现细节
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

#endif // _HEADER_SUPERTONIC_TTS_HPP_