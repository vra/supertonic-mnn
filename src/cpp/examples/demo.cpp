#include "mnn_tts_sdk.hpp"
#include "utils.hpp"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

int write_audio_to_file(const std::string &audio_save_path, const std::vector<int16_t> &audio_data)
{
  ma_result result;
  ma_encoder encoder;

  // 这里设置你的音频数据的格式和音频参数
  ma_encoder_config encoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_s16, 1, 44100);

  result = ma_encoder_init_file(audio_save_path.c_str(), &encoderConfig, &encoder);
  if (result != MA_SUCCESS)
  {
    return -1; // 不能初始化编码器
  }

  // 将音频数据写入到 WAV 文件
  result = ma_encoder_write_pcm_frames(&encoder, audio_data.data(), audio_data.size(), NULL);
  if (result != MA_SUCCESS)
  {
    ma_encoder_uninit(&encoder);
    return -2; // 不能将数据写入文件
  }

  // 清理编码器
  ma_encoder_uninit(&encoder);

  return 0;
}

int main()
{
  MNNITTSLogger::GetInstance().SetLogLevel(TRACE);

  std::string config_folder = "/Users/ws/ws/projects/sdk/pods/PixelAI_TTS_Package/PixelAI_TTS_SDK/cpp/tts_models_private/";

  auto t0 = clk::now();
  auto sdk = MNNTTSSDK(config_folder);
  auto t1 = clk::now();
  PLOG(INFO, "TTS 初始化成功");

  auto d1 = std::chrono::duration_cast<ms>(t1 - t0);
  std::cout << "load total tmecost: " << (double)d1.count() << " ms" << std::endl;

  std::vector<std::string> test_cases = {
      "很高兴遇见你，你叫什么名字呢?"};
  for (int i = 0; i < test_cases.size(); i++)
  {
    auto [sr, audio] = sdk.Process(test_cases[i]);
    std::vector<float> float_audio;
    for (auto &a : audio)
    {
      float_audio.push_back(a / 32767.0f);
    }
    write_audio_to_file("output_" + std::to_string(i) + ".wav", audio);
  }
  return 0;
}