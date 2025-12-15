/**
 * @file mnn_supertonic_tts_impl.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief MNN C++ 版本的Supertonic TTS实现类
 */

#ifndef _HEADER_MNN_SUPERTONIC_TTS_IMPL_H_
#define _HEADER_MNN_SUPERTONIC_TTS_IMPL_H_

#include "mnn_tts_impl_base.hpp"
#include "mnn_tts_logger.hpp"
#include "supertonic.hpp"
#include "supertonic/mnn_duration_predictor.hpp"
#include "supertonic/mnn_estimator.hpp"
#include "supertonic/mnn_text_encoder.hpp"
#include "supertonic/mnn_vocoder.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace MNN::Express;

class MNNSupertonicTTSImpl : public MNNTTSImplBase {
public:
  MNNSupertonicTTSImpl(const std::string &models_dir,
                       const std::string &precision_dir,
                       const std::string &cache_dir);

  // 文本转语音核心接口
  std::tuple<std::vector<float>, int, float>
  synthesize(const std::string &text, const VoiceStyle &voice_style, int steps,
             float speed) override;

  // 兼容基类接口
  std::tuple<int, Audio> Process(const std::string &text) override;

private:
  // 成员变量
  std::string models_dir_;
  std::string precision_dir_;
  std::string cache_dir_;

  // Config values
  int sample_rate_;
  int base_chunk_size_;
  int chunk_compress_factor_;
  int ldim_;

  // 文本处理器类
  class TextProcessor {
  public:
    TextProcessor(const std::string &indexer_path);
    std::vector<int> encode(const std::string &text);
    std::string decode(const std::vector<int> &indices);

  private:
    std::map<uint16_t, int> unicode_to_index_;
    std::map<int, uint16_t> index_to_unicode_;
  };

  // 组件
  std::unique_ptr<TextProcessor> text_processor_;

  // 专用模型推理器
  std::unique_ptr<MNNDurationPredictor> dp_predictor_; // Duration Predictor
  std::unique_ptr<MNNTextEncoder> te_encoder_;         // Text Encoder
  std::unique_ptr<MNNVectorEstimator> ve_estimator_;   // Vector Estimator
  std::unique_ptr<MNNVocoder> vc_vocoder_;             // Vocoder

  // 初始化函数
  void initializeModels();

  // 核心处理函数
  std::vector<float>
  predictDuration(const std::vector<int> &text_ids,
                  const std::vector<std::vector<float>> &style_dp,
                  const std::vector<float> &text_mask);

  std::vector<float>
  encodeText(const std::vector<int> &text_ids,
             const std::vector<std::vector<float>> &style_ttl,
             const std::vector<float> &text_mask);

  std::vector<float> estimateVector(const std::vector<float> &noisy_latent,
                                    const std::vector<float> &text_emb,
                                    const std::vector<float> &style_ttl,
                                    const std::vector<int> &latent_mask,
                                    const std::vector<float> &text_mask,
                                    int current_step, int total_step);

  std::vector<float> vocode(const std::vector<float> &latent);
};

#endif // _HEADER_MNN_SUPERTONIC_TTS_IMPL_H_