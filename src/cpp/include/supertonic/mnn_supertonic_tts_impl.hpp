#ifndef _HEADER_MNN_SUPERTONIC_TTS_IMPL_H_
#define _HEADER_MNN_SUPERTONIC_TTS_IMPL_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "mnn_tts_impl_base.hpp"
#include "mnn_tts_logger.hpp"
#include "supertonic.hpp"

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>

/**
 * @brief MNN C++ implementation of Supertonic TTS.
 *
 * This class handles the complete TTS pipeline including:
 * 1. Text Processing (Normalization, cleaning, encoding)
 * 2. MNN Model Inference (Duration Predictor, Text Encoder, Vector Estimator,
 * Vocoder)
 * 3. Audio Synthesis
 */
class MNNSupertonicTTSImpl : public MNNTTSImplBase {
public:
  MNNSupertonicTTSImpl(const std::string &models_dir,
                       const std::string &precision_dir,
                       const std::string &cache_dir);

  // Core Synthesis Interface
  // Note: VoiceStyle is defined in 'supertonic.hpp'
  std::tuple<std::vector<float>, int, float>
  synthesize(const std::string &text, const VoiceStyle &voice_style, int steps,
             float speed) override;

  // Base Class Interface
  std::tuple<int, Audio> Process(const std::string &text) override;

private:
  // --- Configuration ---
  std::string models_dir_;
  std::string precision_dir_;
  std::string cache_dir_;
  int sample_rate_;
  int base_chunk_size_;
  int chunk_compress_factor_;
  int ldim_;

  // --- MNN Runtime ---
  std::shared_ptr<MNN::Express::Executor> executor_;

  // --- MNN Modules ---
  std::shared_ptr<MNN::Express::Module> dp_module_; // Duration Predictor
  std::shared_ptr<MNN::Express::Module> te_module_; // Text Encoder
  std::shared_ptr<MNN::Express::Module> ve_module_; // Vector Estimator
  std::shared_ptr<MNN::Express::Module> vc_module_; // Vocoder

  // --- Internal Text Processor ---
  class TextProcessor {
  public:
    TextProcessor(const std::string &indexer_path);
    std::vector<int> encode(const std::string &text);
    std::string decode(const std::vector<int> &indices);

  private:
    std::map<uint16_t, int> unicode_to_index_;
    std::map<int, uint16_t> index_to_unicode_;
  };
  std::unique_ptr<TextProcessor> text_processor_;

  // --- Private Helper Methods ---

  // Initialize all MNN models
  void initializeModels();

  // Predict phoneme durations
  std::vector<float>
  predictDuration(const std::vector<int> &text_ids,
                  const std::vector<std::vector<float>> &style_dp,
                  const std::vector<float> &text_mask);

  // Encode text indices into embedding using Style Vector
  std::vector<float>
  encodeText(const std::vector<int> &text_ids,
             const std::vector<std::vector<float>> &style_ttl,
             const std::vector<float> &text_mask);

  // Estimate audio vector (Flow Matching Step)
  std::vector<float> estimateVector(const std::vector<float> &noisy_latent,
                                    const std::vector<float> &text_emb,
                                    const std::vector<float> &style_ttl_flat,
                                    const std::vector<int> &latent_mask,
                                    const std::vector<float> &text_mask,
                                    int current_step, int total_step);

  // Vocode latent vector to audio
  std::vector<float> vocode(const std::vector<float> &latent);
};

#endif // _HEADER_MNN_SUPERTONIC_TTS_IMPL_H_