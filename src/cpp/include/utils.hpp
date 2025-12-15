#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

// 工具函数声明

// 向量展平
template<typename T>
std::vector<T> flattenVector(const std::vector<std::vector<T>>& vec2d) {
    std::vector<T> result;
    for (const auto& row : vec2d) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

// 文件存在检查
namespace utils {
    inline bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
}

// WAV文件写入函数
bool writeWavFile(const std::string& filename, 
                  const std::vector<float>& audio_data, 
                  int sample_rate);

// 字符串连接函数
std::string joinStrings(const std::vector<std::string>& strings, const std::string& delimiter);
