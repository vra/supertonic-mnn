/**
 * @file supertonic.cpp
 * @brief Supertonic TTS C++ SDK 实现文件
 */

#include "supertonic/supertonic.hpp"
#include "supertonic/mnn_supertonic_tts_impl.hpp"
#include "mnn_tts_config.hpp"
#include "mnn_tts_logger.hpp"
#include "utils.hpp"

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>

using json = nlohmann::json;
using namespace MNN::Express;

// 内部实现类
class SupertonicTTS::Impl
{
public:
    explicit Impl(const TTSConfig &config) : config_(config)
    {
        // 初始化日志系统
        MNNITTSLogger::GetInstance().SetLogLevel(INFO);

        // 加载TTS引擎
        loadTTSEngine();
    }

    std::tuple<std::vector<float>, int, float> synthesize(
        const std::string &text,
        const std::string &voice,
        int steps,
        float speed,
        const std::string &language)
    {

        // 文本预处理
        std::string processed_text = preprocessText(text, language);
        // 检查是否有所需的语音风格
        if (voice_styles_.find(voice) == voice_styles_.end())
        {
            throw std::runtime_error("Voice style not found: " + voice);
        }

        // 调用实现类的synthesize方法
        auto [audio_data, sample_rate, rtf] = tts_impl_->synthesize(processed_text, voice_styles_[voice], steps, speed);

        return std::make_tuple(audio_data, sample_rate, rtf);
    }

    static bool save(const std::string &filename,
                     const std::vector<float> &audio_data,
                     int sample_rate)
    {
        return writeWavFile(filename, audio_data, sample_rate);
    }

private:
    TTSConfig config_;
    std::unique_ptr<MNNTTSImplBase> tts_impl_;
    std::map<std::string, VoiceStyle> voice_styles_;

    void loadTTSEngine()
    {
        PLOG(INFO, "Loading TTS engine...");

        // 展开波浪号路径
        std::string model_dir = config_.model_dir;
        if (model_dir.substr(0, 2) == "~/")
        {
            const char *home_dir = getenv("HOME");
            if (home_dir)
            {
                model_dir = std::string(home_dir) + model_dir.substr(1);
            }
        }

        // 创建Supertonic TTS实现
        tts_impl_ = std::make_unique<MNNSupertonicTTSImpl>(
            model_dir,
            config_.precision,
            model_dir + "/cache");

        // 加载所有可用的语音风格
        loadVoiceStyles(model_dir);

        PLOG(INFO, "TTS engine loaded successfully");
    }

    void loadVoiceStyles(const std::string &model_dir)
    {
        PLOG(INFO, "Loading voice styles...");

        std::vector<std::string> voice_ids = {
            "M1",
            "M2",
            "F1",
            "F2",
        };
        for (auto &voice_id : voice_ids)
        {
            loadVoiceStyle(model_dir, voice_id);
        }
        PLOG(INFO, "Voice styles loaded successfully");
    }

    void loadVoiceStyle(const std::string &model_dir, const std::string &voice_name)
    {
        try
        {
            std::string style_file_path = model_dir + "/voice_styles/" + voice_name + ".json";

            // 读取style.json
            std::ifstream style_file(style_file_path);
            if (!style_file.is_open())
            {
                PLOG(WARNING, "Failed to open style.json for voice: " + voice_name);
                return;
            }

            json style_json;
            style_file >> style_json;
            style_file.close();

            // 解析ttl和dp向量
            std::vector<std::vector<float>> ttl_vectors, dp_vectors;

            if (style_json.contains("style_ttl") && style_json.contains("style_dp"))
            {
                std::cout << "1" << std::endl;
                // 解析ttl向量
                for (const auto &ttl_item : style_json["style_ttl"]["data"])
                {
                    for (const auto &vec : ttl_item)
                    {
                        std::vector<float> ttl_vector;
                        for (const auto &val : vec)
                        {
                            ttl_vector.push_back(val.get<float>());
                        }
                        ttl_vectors.push_back(ttl_vector);
                    }
                }

                // 解析dp向量
                for (const auto &dp_item : style_json["style_dp"]["data"])
                {
                    for (const auto &vec : dp_item)
                    {
                        std::vector<float> dp_vector;
                        for (const auto &val : vec)
                        {
                            dp_vector.push_back(val.get<float>());
                        }
                        dp_vectors.push_back(dp_vector);
                    }
                }

                // 重塑向量维度以匹配模型期望的形状
                auto ttl_reshaped = reshapeVectors(ttl_vectors, {1, 1, static_cast<int>(ttl_vectors.size()), static_cast<int>(ttl_vectors[0].size())});
                auto dp_reshaped = reshapeVectors(dp_vectors, {1, 1, static_cast<int>(dp_vectors.size()), static_cast<int>(dp_vectors[0].size())});

                // 存储语音风格
                voice_styles_.insert(std::make_pair(voice_name, VoiceStyle(ttl_reshaped, dp_reshaped)));

                PLOG(INFO, "Loaded voice style: " + voice_name);
            }
        }
        catch (const std::exception &e)
        {
            PLOG(ERROR, "Error loading voice style " + voice_name + ": " + e.what());
        }
        for (auto &e : voice_styles_)
        {
            std::cout << e.first << std::endl;
        }
    }

    void loadDefaultVoiceStyles()
    {
        // 创建一些默认的语音风格向量
        std::vector<std::vector<float>> default_ttl = {{0.1f, 0.2f, 0.3f}};
        std::vector<std::vector<float>> default_dp = {{0.4f, 0.5f, 0.6f}};

        auto ttl_reshaped = reshapeVectors(default_ttl, {1, 1, 1, 3});
        auto dp_reshaped = reshapeVectors(default_dp, {1, 1, 1, 3});

        voice_styles_["default"] = VoiceStyle(ttl_reshaped, dp_reshaped);
    }

    std::vector<std::vector<float>> reshapeVectors(
        const std::vector<std::vector<float>> &input,
        const std::vector<int> &shape)
    {

        // 简单的reshape实现，实际应用中可能需要更复杂的逻辑
        return input;
    }

    std::string preprocessText(const std::string &text, const std::string &language)
    {
        // 移除多余的空白字符
        std::string processed = std::regex_replace(text, std::regex("\\s+"), " ");

        // 移除首尾空白字符
        processed = std::regex_replace(processed, std::regex("^\\s+|\\s+$"), "");

        // 根据语言进行特定的预处理
        if (language == "zh")
        {
            // 中文文本预处理
            processed = std::regex_replace(processed, std::regex("[\\.!?]+$"), "。");
        }
        else
        {
            // 英文文本预处理
            processed = std::regex_replace(processed, std::regex("[\\.!?]+$"), ".");
        }

        return processed;
    }
};

// SupertonicTTS公共接口实现
SupertonicTTS::SupertonicTTS(const TTSConfig &config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SupertonicTTS::~SupertonicTTS() = default;

std::tuple<std::vector<float>, int, float> SupertonicTTS::synthesize(
    const std::string &text,
    const std::string &voice,
    int steps,
    float speed,
    const std::string &language)
{
    return pimpl_->synthesize(text, voice, steps, speed, language);
}

bool SupertonicTTS::save(const std::string &filename,
                         const std::vector<float> &audio_data,
                         int sample_rate)
{
    return Impl::save(filename, audio_data, sample_rate);
}