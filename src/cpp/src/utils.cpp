#include "utils.hpp"
#include "mnn_tts_logger.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>

// WAV文件头结构
struct WavHeader {
  // RIFF header
  char riff_header[4] = {'R', 'I', 'F', 'F'};
  uint32_t wav_size = 0;
  char wave_header[4] = {'W', 'A', 'V', 'E'};

  // Format header
  char fmt_header[4] = {'f', 'm', 't', ' '};
  uint32_t fmt_chunk_size = 16;
  uint16_t audio_format = 1; // PCM
  uint16_t num_channels = 1;
  uint32_t sample_rate = 0;
  uint32_t byte_rate = 0;
  uint16_t block_align = 0;
  uint16_t bits_per_sample = 16;

  // Data header
  char data_header[4] = {'d', 'a', 't', 'a'};
  uint32_t data_bytes = 0;
};

bool writeWavFile(const std::string &filename,
                  const std::vector<float> &audio_data, int sample_rate) {
  // 转换浮点数据为16位整数
  std::vector<int16_t> int_data(audio_data.size());
  for (size_t i = 0; i < audio_data.size(); ++i) {
    // 限制范围在 [-1.0, 1.0]
    float sample = std::max(-1.0f, std::min(1.0f, audio_data[i]));
    int_data[i] = static_cast<int16_t>(sample * 32767.0f);
  }

  // 创建WAV头
  WavHeader header;
  header.audio_format = 1;
  header.num_channels = 1;
  header.sample_rate = sample_rate;
  header.byte_rate = sample_rate * 2; // 16 bits = 2 bytes
  header.block_align = 2;             // 16 bits = 2 bytes
  header.bits_per_sample = 16;
  header.data_bytes = static_cast<uint32_t>(int_data.size() * sizeof(int16_t));
  header.wav_size = 36 + header.data_bytes; // 36 is the size of the header

  // 写入文件
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    PLOG(ERROR, "Failed to open file for writing: " + filename);
    return false;
  }

  // 写入头部
  file.write(reinterpret_cast<const char *>(&header), sizeof(header));

  // 写入音频数据
  file.write(reinterpret_cast<const char *>(int_data.data()),
             header.data_bytes);

  file.close();

  PLOG(INFO, "WAV file written successfully: " + filename);
  return true;
}

std::string joinStrings(const std::vector<std::string> &strings,
                        const std::string &delimiter) {
  if (strings.empty()) {
    return "";
  }

  std::string result = strings[0];
  for (size_t i = 1; i < strings.size(); ++i) {
    result += delimiter + strings[i];
  }

  return result;
}