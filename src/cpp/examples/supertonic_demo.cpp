/**
 * @file supertonic_demo.cpp
 * @brief Supertonic TTS C++ SDK Example
 */

#include "mnn_tts_logger.hpp"
#include "mnn_tts_sdk.hpp"
#include "utils.hpp"
#include <chrono>
#include <iostream>

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

int main()
{
  try
  {
    // Set Log Level
    MNNITTSLogger::GetInstance().SetLogLevel(INFO);

    std::string config_dir = "/Users/ws/supertonic-tts-mnn";
    auto t0 = clk::now();
    MNNTTSSDK tts(config_dir);
    auto t1 = clk::now();

    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);
    PLOG(INFO, "TTS engine initialization time: " + std::to_string(d1.count()) +
                   " ms");

    // Test Text
    std::string text =
        "It’s nice to have shortcuts for commands you use often. To have "
        "normalized names for installing dependencies, running the code, or "
        "starting the tests across all your projects.";
    std::string voice = "F1";

    // Text to Speech
    auto [sample_rate, audio_data] = tts.Process(text);
    std::cout <<"==> sample rate: " << sample_rate << std::endl;

    std::string output_file = "output.wav";
    // tts.save(audio_data, output_file);
    std::vector<float> audio_fp32;
    for (auto& e: audio_data) {
      audio_fp32.push_back(float(e)/32768.0f);
    }
    writeWavFile(output_file, audio_fp32, 44100);
    return 0;
  }
  catch (const std::exception &e)
  {
    PLOG(ERROR, "Error: " + std::string(e.what()));
    return 1;
  }
}