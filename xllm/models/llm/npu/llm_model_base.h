/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/npu/loader/base_manual_loader.h"
#include "core/layers/npu/loader/rolling_load_manager.h"
#include "core/layers/npu/loader/rolling_weight_buffer.h"
#include "core/layers/npu/npu_base_layer.h"
#include "core/layers/npu/npu_block_copy_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_pos_embedding_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "models/model_registry.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"

namespace xllm {

namespace {
std::atomic<int64_t> g_layer_exec_profiled_batches{0};
std::atomic<int64_t> g_layer_exec_profiled_exec_us{0};
std::atomic<int64_t> g_layer_exec_profiled_interval_us{0};
std::atomic<int64_t> g_layer_exec_profiled_forward_us{0};
std::atomic<int64_t> g_layer_exec_profiled_chunk_exec_us{0};
std::atomic<int64_t> g_layer_exec_profiled_chunk_idle_us{0};
std::atomic<int64_t> g_layer_exec_profiled_chunk_count{0};
std::atomic<int64_t> g_layer_exec_profiled_chunk_nonzero_count{0};
std::atomic<int64_t> g_layer_exec_profiled_forward_us_max{0};
std::atomic<int64_t> g_layer_exec_profiled_chunk_idle_us_max{0};
std::atomic<int64_t> g_layer_exec_profiled_all_chunk_idle_us_max{0};
std::mutex g_layer_exec_profiled_chunk_totals_mutex;
std::vector<int64_t> g_layer_exec_profiled_chunk_idle_us_sum_list;
std::vector<int64_t> g_layer_exec_profiled_chunk_sample_count_list;
std::vector<int64_t> g_layer_exec_profiled_chunk_idle_us_max_list;
std::vector<std::priority_queue<int64_t>>
    g_layer_exec_profiled_chunk_idle_lower_heaps;
std::vector<
    std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>>>
    g_layer_exec_profiled_chunk_idle_upper_heaps;
std::mutex g_layer_exec_profiled_batch_idle_median_mutex;
std::priority_queue<int64_t> g_layer_exec_profiled_batch_idle_lower;
std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>>
    g_layer_exec_profiled_batch_idle_upper;
std::mutex g_layer_exec_profiled_batch_idle_samples_mutex;
std::vector<int64_t> g_layer_exec_profiled_batch_idle_samples;
std::mutex g_layer_exec_profiled_batch_forward_samples_mutex;
std::vector<int64_t> g_layer_exec_profiled_batch_forward_samples;
std::mutex g_layer_exec_profiled_all_chunk_idle_median_mutex;
std::priority_queue<int64_t> g_layer_exec_profiled_all_chunk_idle_lower;
std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>>
    g_layer_exec_profiled_all_chunk_idle_upper;
std::mutex g_layer_exec_profiled_all_chunk_idle_samples_mutex;
std::vector<int64_t> g_layer_exec_profiled_all_chunk_idle_samples;

double get_sorted_quantile_us(const std::vector<int64_t>& sorted_values,
                              double q) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double clamped_q = std::max(0.0, std::min(1.0, q));
  const double pos = (sorted_values.size() - 1) * clamped_q;
  const size_t low_idx = static_cast<size_t>(std::floor(pos));
  const size_t high_idx = static_cast<size_t>(std::ceil(pos));
  if (low_idx == high_idx) {
    return static_cast<double>(sorted_values[low_idx]);
  }
  const double low_v = static_cast<double>(sorted_values[low_idx]);
  const double high_v = static_cast<double>(sorted_values[high_idx]);
  const double frac = pos - low_idx;
  return low_v + frac * (high_v - low_v);
}
}  // namespace

template <typename DecoderType>
class LlmDecoderLayerImplBase : public torch::nn::Module {
 public:
  LlmDecoderLayerImplBase(const ModelContext& context,
                          const int32_t layer_id = -1)
      : layer_id_(layer_id) {
    CHECK(layer_id_ >= 0) << "layer_id must be >= 0, but got " << layer_id_;
    // register submodules
    decoder_layer_ = register_module("decoder_layer", DecoderType(context));
    block_copy_ = register_module("block_copy", layer::NpuBlockCopy(context));
  }

  virtual torch::Tensor forward(torch::Tensor& x,
                                torch::Tensor& cos_pos,
                                torch::Tensor& sin_pos,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                aclrtEvent* event,
                                std::atomic<bool>* event_flag) {
    if (input_params.src_block_indices.numel() > 0) {
      block_copy_(kv_cache.get_k_cache(),
                  kv_cache.get_v_cache(),
                  input_params.src_block_indices,
                  input_params.dst_block_indices,
                  input_params.cum_sum,
                  0);
    }

    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag,
                          layer_id_);
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights();
  }
  virtual void merge_loaded_weights() {
    decoder_layer_->merge_loaded_weights();
    block_copy_->merge_loaded_weights();
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  virtual void merge_and_move_pinned_host() {
    decoder_layer_->merge_and_move_pinned_host();
    block_copy_->merge_loaded_weights();
  }

  virtual void free_weights() { decoder_layer_->free_weights(); }

  virtual void reload_weights() { decoder_layer_->reload_weights(); }

  virtual void reload_weights_from_device() {
    decoder_layer_->reload_weights_from_device();
  }

  virtual layer::BaseManualLoader* get_manual_loader() {
    return decoder_layer_->get_manual_loader();
  }

  virtual void refresh_rolling_weights() {
    decoder_layer_->refresh_rolling_weights();
  }

 private:
  DecoderType decoder_layer_{nullptr};
  layer::NpuBlockCopy block_copy_{nullptr};
  int32_t layer_id_;
};

template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 public:
  // mode type: qwen2, qwen3 .etc
  LlmModelImplBase(const std::string& model_type, const ModelArgs& args)
      : model_type_(model_type) {
    InterruptionBus::get_instance().subscribe([this](bool interrupted) {
      this->layer_forward_interrupted_ = interrupted;
    });
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return npu_embed_tokens_(input_ids, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.input_embedding;
    // test
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }

    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        auto sections = mrope_section_;
        sections.insert(sections.end(), sections.begin(), sections.end());

        auto vec = x.split(sections, -1);
        std::vector<torch::Tensor> selects;
        selects.reserve(vec.size());

        for (int64_t i = 0; i < vec.size(); ++i) {
          auto m = vec[i];
          selects.push_back(m[i % mrope_section_.size()]);
        }
        return torch::cat(selects, -1);
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor attn_mask;
    max_seq_len_ = FLAGS_enable_chunked_prefill
                       ? std::max(input_params.kv_max_seq_len, max_seq_len_)
                       : 128;
    if (model_type_ == "qwen2") {
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
    } else {
      if (FLAGS_enable_chunked_prefill) {
        int num_sequences = input_params.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int j = 0; j < num_sequences; j++) {
            auto mask =
                attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                           input_params.kv_seq_lens_vec[j],
                                           max_seq_len_,
                                           cos_pos.dtype().toScalarType(),
                                           cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    RollingLayerGuard rolling_guard(rolling_mgr_);

    const bool enable_layer_exec_profile = FLAGS_enable_layer_exec_profile;
    const auto batch_profile_start = std::chrono::steady_clock::now();
    auto prev_layer_end = batch_profile_start;
    std::vector<int64_t> layer_exec_us_list;
    std::vector<int64_t> layer_interval_us_list;
    std::vector<int64_t> chunk_exec_us_list;
    std::vector<int64_t> chunk_idle_us_list;
    int64_t batch_exec_us = 0;
    int64_t batch_interval_us = 0;
    int64_t batch_forward_us = 0;
    int64_t batch_chunk_exec_us = 0;
    int64_t batch_chunk_idle_us = 0;
    int64_t batch_chunk_count = 0;
    int64_t batch_chunk_nonzero_count = 0;
    int64_t max_layer_exec_us = 0;
    int64_t max_layer_interval_us = 0;
    int64_t max_chunk_exec_us = 0;
    int64_t max_chunk_idle_us = 0;
    int64_t max_layer_exec_idx = -1;
    int64_t max_layer_interval_idx = -1;
    int64_t max_chunk_exec_idx = -1;
    int64_t max_chunk_idle_idx = -1;
    uint32_t chunk_layers = 0;

#if defined(USE_NPU)
    bool enable_compute_chunk_profile = false;
    aclrtStream compute_stream = nullptr;
    aclrtEvent batch_anchor_event = nullptr;
    std::vector<aclrtEvent> chunk_start_events;
    std::vector<aclrtEvent> chunk_end_events;
    auto destroy_event = [](aclrtEvent& event) {
      if (event != nullptr) {
        aclrtDestroyEvent(event);
        event = nullptr;
      }
    };
    auto cleanup_chunk_events = [&]() {
      destroy_event(batch_anchor_event);
      for (auto& event : chunk_start_events) {
        destroy_event(event);
      }
      for (auto& event : chunk_end_events) {
        destroy_event(event);
      }
    };
    bool chunk_profile_record_error_logged = false;
    bool chunk_profile_elapsed_error_logged = false;
#endif

    if (enable_layer_exec_profile) {
      layer_exec_us_list.assign(layers_.size(), 0);
      layer_interval_us_list.assign(layers_.size(), 0);
      // Use FLAGS_layers_wise_copy_batchs as target chunk count, then derive
      // layers per chunk by dividing hidden-layer count by that chunk count.
      const uint32_t target_chunk_count =
          std::max<uint32_t>(1, FLAGS_layers_wise_copy_batchs);
      chunk_layers = layers_.size() / target_chunk_count;
      if (chunk_layers == 0) {
        chunk_layers = 1;
      }
      const size_t chunk_count =
          (layers_.size() + chunk_layers - 1) / chunk_layers;
      chunk_exec_us_list.assign(chunk_count, 0);
      chunk_idle_us_list.assign(chunk_count, 0);

#if defined(USE_NPU)
      if (chunk_count > 0) {
        compute_stream =
            c10_npu::getCurrentNPUStream(h.device().index()).stream();
        chunk_start_events.assign(chunk_count, nullptr);
        chunk_end_events.assign(chunk_count, nullptr);
        bool create_success = true;
        if (aclrtCreateEvent(&batch_anchor_event) != ACL_SUCCESS) {
          create_success = false;
        }
        for (size_t chunk_idx = 0; create_success && chunk_idx < chunk_count;
             ++chunk_idx) {
          if (aclrtCreateEvent(&chunk_start_events[chunk_idx]) != ACL_SUCCESS) {
            create_success = false;
            break;
          }
          if (aclrtCreateEvent(&chunk_end_events[chunk_idx]) != ACL_SUCCESS) {
            create_success = false;
            break;
          }
        }
        aclError anchor_ret = ACL_ERROR_NONE;
        if (create_success) {
          anchor_ret = aclrtRecordEvent(batch_anchor_event, compute_stream);
        }
        if (create_success && anchor_ret == ACL_SUCCESS) {
          enable_compute_chunk_profile = true;
        } else {
          cleanup_chunk_events();
          LOG(WARNING) << "Failed to initialize chunk compute stream profile "
                       << "events, fall back to host-side layer profile only. "
                       << "create_success=" << create_success
                       << ", anchor_ret=" << anchor_ret
                       << ", stream=" << static_cast<void*>(compute_stream);
        }
      }
#endif
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
#if defined(USE_NPU)
        cleanup_chunk_events();
#endif
        return ModelOutput();
      }

      auto& layer = layers_[i];

      if (layer_forward_interrupted_) {
        LOG(INFO) << "Forward interrupted at layer: " << i;
#if defined(USE_NPU)
        cleanup_chunk_events();
#endif
        return ModelOutput();
      }
      const int32_t layer_index = i;
      rolling_guard.before_layer(layer_index);

#if defined(USE_NPU)
      if (enable_compute_chunk_profile && chunk_layers > 0 &&
          i % chunk_layers == 0) {
        const size_t chunk_idx = i / chunk_layers;
        const auto ret =
            aclrtRecordEvent(chunk_start_events[chunk_idx], compute_stream);
        if (ret != ACL_SUCCESS && !chunk_profile_record_error_logged) {
          chunk_profile_record_error_logged = true;
          LOG(WARNING)
              << "[chunk_profile_debug] record chunk start event failed: "
              << "ret=" << ret << ", chunk_idx=" << chunk_idx
              << ", stream=" << static_cast<void*>(compute_stream);
        }
      }
#endif

      const auto layer_start = std::chrono::steady_clock::now();
      if (enable_layer_exec_profile) {
        const int64_t interval_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                layer_start - prev_layer_end)
                .count();
        batch_interval_us += interval_us;
        layer_interval_us_list[i] = interval_us;
        if (interval_us > max_layer_interval_us) {
          max_layer_interval_us = interval_us;
          max_layer_interval_idx = static_cast<int64_t>(i);
        }
      }

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);

      if (enable_layer_exec_profile) {
        const auto layer_end = std::chrono::steady_clock::now();
        const int64_t exec_us =
            std::chrono::duration_cast<std::chrono::microseconds>(layer_end -
                                                                  layer_start)
                .count();
        batch_exec_us += exec_us;
        layer_exec_us_list[i] = exec_us;
        if (exec_us > max_layer_exec_us) {
          max_layer_exec_us = exec_us;
          max_layer_exec_idx = static_cast<int64_t>(i);
        }
        prev_layer_end = layer_end;
      }

#if defined(USE_NPU)
      if (enable_compute_chunk_profile && chunk_layers > 0) {
        const size_t chunk_idx = i / chunk_layers;
        const size_t chunk_end_layer =
            std::min(layers_.size(),
                     static_cast<size_t>((chunk_idx + 1) * chunk_layers));
        if (i + 1 == chunk_end_layer) {
          const auto ret =
              aclrtRecordEvent(chunk_end_events[chunk_idx], compute_stream);
          if (ret != ACL_SUCCESS && !chunk_profile_record_error_logged) {
            chunk_profile_record_error_logged = true;
            LOG(WARNING)
                << "[chunk_profile_debug] record chunk end event failed: "
                << "ret=" << ret << ", chunk_idx=" << chunk_idx
                << ", stream=" << static_cast<void*>(compute_stream);
          }
        }
      }
#endif

      rolling_guard.after_layer(layer_index);
    }

    if (enable_layer_exec_profile) {
#if defined(USE_NPU)
      if (enable_compute_chunk_profile && !chunk_end_events.empty()) {
        const auto ret =
            aclrtSynchronizeEventWithTimeout(chunk_end_events.back(), -1);
        if (ret != ACL_SUCCESS) {
          LOG(WARNING) << "Synchronize chunk end event failed: " << ret;
        } else {
          for (size_t chunk_idx = 0; chunk_idx < chunk_end_events.size();
               ++chunk_idx) {
            float exec_ms = 0.0f;
            const auto exec_elapsed_ret =
                aclrtEventElapsedTime(&exec_ms,
                                      chunk_start_events[chunk_idx],
                                      chunk_end_events[chunk_idx]);
            if (exec_elapsed_ret == ACL_SUCCESS) {
              const int64_t exec_us = static_cast<int64_t>(exec_ms * 1000.0f);
              chunk_exec_us_list[chunk_idx] = exec_us;
              batch_chunk_exec_us += exec_us;
              if (exec_us > max_chunk_exec_us) {
                max_chunk_exec_us = exec_us;
                max_chunk_exec_idx = static_cast<int64_t>(chunk_idx);
              }
            } else if (!chunk_profile_elapsed_error_logged) {
              chunk_profile_elapsed_error_logged = true;
              LOG(WARNING)
                  << "[chunk_profile_debug] elapsed chunk exec failed: "
                  << "ret=" << exec_elapsed_ret << ", chunk_idx=" << chunk_idx
                  << ", event_stream=" << static_cast<void*>(compute_stream);
            }
            if (chunk_idx == 0) {
              float idle_ms = 0.0f;
              const auto idle_elapsed_ret = aclrtEventElapsedTime(
                  &idle_ms, batch_anchor_event, chunk_start_events[chunk_idx]);
              if (idle_elapsed_ret == ACL_SUCCESS) {
                const int64_t idle_us = static_cast<int64_t>(idle_ms * 1000.0f);
                chunk_idle_us_list[chunk_idx] = idle_us;
                batch_chunk_idle_us += idle_us;
                if (idle_us > max_chunk_idle_us) {
                  max_chunk_idle_us = idle_us;
                  max_chunk_idle_idx = static_cast<int64_t>(chunk_idx);
                }
              } else if (!chunk_profile_elapsed_error_logged) {
                chunk_profile_elapsed_error_logged = true;
                LOG(WARNING)
                    << "[chunk_profile_debug] elapsed initial chunk idle "
                    << "failed: ret=" << idle_elapsed_ret
                    << ", chunk_idx=" << chunk_idx
                    << ", event_stream=" << static_cast<void*>(compute_stream);
              }
            } else {
              float idle_ms = 0.0f;
              const auto idle_elapsed_ret =
                  aclrtEventElapsedTime(&idle_ms,
                                        chunk_end_events[chunk_idx - 1],
                                        chunk_start_events[chunk_idx]);
              if (idle_elapsed_ret == ACL_SUCCESS) {
                const int64_t idle_us = static_cast<int64_t>(idle_ms * 1000.0f);
                chunk_idle_us_list[chunk_idx] = idle_us;
                batch_chunk_idle_us += idle_us;
                if (idle_us > max_chunk_idle_us) {
                  max_chunk_idle_us = idle_us;
                  max_chunk_idle_idx = static_cast<int64_t>(chunk_idx);
                }
              } else if (!chunk_profile_elapsed_error_logged) {
                chunk_profile_elapsed_error_logged = true;
                LOG(WARNING)
                    << "[chunk_profile_debug] elapsed chunk idle failed: "
                    << "ret=" << idle_elapsed_ret << ", chunk_idx=" << chunk_idx
                    << ", event_stream=" << static_cast<void*>(compute_stream);
              }
            }
          }
        }
      }
      cleanup_chunk_events();
#endif

      batch_forward_us = batch_chunk_exec_us + batch_chunk_idle_us;
      if (batch_forward_us <= 0) {
        batch_forward_us = batch_exec_us + batch_interval_us;
      }

      batch_chunk_count = static_cast<int64_t>(chunk_idle_us_list.size());
      for (const auto idle_us : chunk_idle_us_list) {
        if (idle_us != 0) {
          batch_chunk_nonzero_count += 1;
        }
      }
      {
        std::lock_guard<std::mutex> lock(
            g_layer_exec_profiled_chunk_totals_mutex);
        if (g_layer_exec_profiled_chunk_idle_us_sum_list.size() <
            chunk_idle_us_list.size()) {
          g_layer_exec_profiled_chunk_idle_us_sum_list.resize(
              chunk_idle_us_list.size(), 0);
          g_layer_exec_profiled_chunk_sample_count_list.resize(
              chunk_idle_us_list.size(), 0);
          g_layer_exec_profiled_chunk_idle_us_max_list.resize(
              chunk_idle_us_list.size(), 0);
          g_layer_exec_profiled_chunk_idle_lower_heaps.resize(
              chunk_idle_us_list.size());
          g_layer_exec_profiled_chunk_idle_upper_heaps.resize(
              chunk_idle_us_list.size());
        }
        for (size_t chunk_idx = 0; chunk_idx < chunk_idle_us_list.size();
             ++chunk_idx) {
          const int64_t idle_us = chunk_idle_us_list[chunk_idx];
          g_layer_exec_profiled_chunk_idle_us_sum_list[chunk_idx] += idle_us;
          g_layer_exec_profiled_chunk_sample_count_list[chunk_idx] += 1;
          if (idle_us >
              g_layer_exec_profiled_chunk_idle_us_max_list[chunk_idx]) {
            g_layer_exec_profiled_chunk_idle_us_max_list[chunk_idx] = idle_us;
          }
          auto& lower_heap =
              g_layer_exec_profiled_chunk_idle_lower_heaps[chunk_idx];
          auto& upper_heap =
              g_layer_exec_profiled_chunk_idle_upper_heaps[chunk_idx];
          if (lower_heap.empty() || idle_us <= lower_heap.top()) {
            lower_heap.push(idle_us);
          } else {
            upper_heap.push(idle_us);
          }
          if (lower_heap.size() > upper_heap.size() + 1) {
            upper_heap.push(lower_heap.top());
            lower_heap.pop();
          } else if (upper_heap.size() > lower_heap.size()) {
            lower_heap.push(upper_heap.top());
            upper_heap.pop();
          }
        }
      }
      int64_t prev_max_all_chunk_idle_us =
          g_layer_exec_profiled_all_chunk_idle_us_max.load(
              std::memory_order_relaxed);
      while (prev_max_all_chunk_idle_us < max_chunk_idle_us &&
             !g_layer_exec_profiled_all_chunk_idle_us_max.compare_exchange_weak(
                 prev_max_all_chunk_idle_us,
                 max_chunk_idle_us,
                 std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }

      const int64_t total_exec_us =
          g_layer_exec_profiled_exec_us.fetch_add(batch_exec_us,
                                                  std::memory_order_relaxed) +
          batch_exec_us;
      const int64_t total_interval_us =
          g_layer_exec_profiled_interval_us.fetch_add(
              batch_interval_us, std::memory_order_relaxed) +
          batch_interval_us;
      const int64_t total_forward_us =
          g_layer_exec_profiled_forward_us.fetch_add(
              batch_forward_us, std::memory_order_relaxed) +
          batch_forward_us;
      const int64_t total_chunk_exec_us =
          g_layer_exec_profiled_chunk_exec_us.fetch_add(
              batch_chunk_exec_us, std::memory_order_relaxed) +
          batch_chunk_exec_us;
      const int64_t total_chunk_idle_us =
          g_layer_exec_profiled_chunk_idle_us.fetch_add(
              batch_chunk_idle_us, std::memory_order_relaxed) +
          batch_chunk_idle_us;
      const int64_t total_chunk_count =
          g_layer_exec_profiled_chunk_count.fetch_add(
              batch_chunk_count, std::memory_order_relaxed) +
          batch_chunk_count;
      const int64_t total_chunk_nonzero_count =
          g_layer_exec_profiled_chunk_nonzero_count.fetch_add(
              batch_chunk_nonzero_count, std::memory_order_relaxed) +
          batch_chunk_nonzero_count;
      int64_t prev_max_chunk_idle_us =
          g_layer_exec_profiled_chunk_idle_us_max.load(
              std::memory_order_relaxed);
      while (prev_max_chunk_idle_us < batch_chunk_idle_us &&
             !g_layer_exec_profiled_chunk_idle_us_max.compare_exchange_weak(
                 prev_max_chunk_idle_us,
                 batch_chunk_idle_us,
                 std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }
      int64_t prev_max_forward_us =
          g_layer_exec_profiled_forward_us_max.load(std::memory_order_relaxed);
      while (prev_max_forward_us < batch_forward_us &&
             !g_layer_exec_profiled_forward_us_max.compare_exchange_weak(
                 prev_max_forward_us,
                 batch_forward_us,
                 std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }
      // Track exact running median (p50) of batch_chunk_idle_us across all
      // profiled steps.
      {
        std::lock_guard<std::mutex> lock(
            g_layer_exec_profiled_batch_idle_median_mutex);
        if (g_layer_exec_profiled_batch_idle_lower.empty() ||
            batch_chunk_idle_us <=
                g_layer_exec_profiled_batch_idle_lower.top()) {
          g_layer_exec_profiled_batch_idle_lower.push(batch_chunk_idle_us);
        } else {
          g_layer_exec_profiled_batch_idle_upper.push(batch_chunk_idle_us);
        }
        if (g_layer_exec_profiled_batch_idle_lower.size() >
            g_layer_exec_profiled_batch_idle_upper.size() + 1) {
          g_layer_exec_profiled_batch_idle_upper.push(
              g_layer_exec_profiled_batch_idle_lower.top());
          g_layer_exec_profiled_batch_idle_lower.pop();
        } else if (g_layer_exec_profiled_batch_idle_upper.size() >
                   g_layer_exec_profiled_batch_idle_lower.size()) {
          g_layer_exec_profiled_batch_idle_lower.push(
              g_layer_exec_profiled_batch_idle_upper.top());
          g_layer_exec_profiled_batch_idle_upper.pop();
        }
      }
      {
        std::lock_guard<std::mutex> lock(
            g_layer_exec_profiled_batch_idle_samples_mutex);
        g_layer_exec_profiled_batch_idle_samples.push_back(batch_chunk_idle_us);
      }
      {
        std::lock_guard<std::mutex> lock(
            g_layer_exec_profiled_batch_forward_samples_mutex);
        g_layer_exec_profiled_batch_forward_samples.push_back(batch_forward_us);
      }
      // Track exact running median (p50) of all chunk idle values across all
      // profiled steps.
      {
        std::lock_guard<std::mutex> lock(
            g_layer_exec_profiled_all_chunk_idle_median_mutex);
        for (const auto idle_us : chunk_idle_us_list) {
          if (g_layer_exec_profiled_all_chunk_idle_lower.empty() ||
              idle_us <= g_layer_exec_profiled_all_chunk_idle_lower.top()) {
            g_layer_exec_profiled_all_chunk_idle_lower.push(idle_us);
          } else {
            g_layer_exec_profiled_all_chunk_idle_upper.push(idle_us);
          }
          if (g_layer_exec_profiled_all_chunk_idle_lower.size() >
              g_layer_exec_profiled_all_chunk_idle_upper.size() + 1) {
            g_layer_exec_profiled_all_chunk_idle_upper.push(
                g_layer_exec_profiled_all_chunk_idle_lower.top());
            g_layer_exec_profiled_all_chunk_idle_lower.pop();
          } else if (g_layer_exec_profiled_all_chunk_idle_upper.size() >
                     g_layer_exec_profiled_all_chunk_idle_lower.size()) {
            g_layer_exec_profiled_all_chunk_idle_lower.push(
                g_layer_exec_profiled_all_chunk_idle_upper.top());
            g_layer_exec_profiled_all_chunk_idle_upper.pop();
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(
            g_layer_exec_profiled_all_chunk_idle_samples_mutex);
        g_layer_exec_profiled_all_chunk_idle_samples.insert(
            g_layer_exec_profiled_all_chunk_idle_samples.end(),
            chunk_idle_us_list.begin(),
            chunk_idle_us_list.end());
      }
      const int64_t profiled_batches = g_layer_exec_profiled_batches.fetch_add(
                                           1, std::memory_order_relaxed) +
                                       1;
      const int64_t log_interval =
          std::max<int64_t>(1, FLAGS_layer_exec_profile_log_interval);
      if (profiled_batches % log_interval == 0) {
        const int64_t avg_exec_us = total_exec_us / profiled_batches;
        const int64_t avg_interval_us = total_interval_us / profiled_batches;
        const int64_t avg_forward_us = total_forward_us / profiled_batches;
        const int64_t avg_chunk_exec_us =
            total_chunk_exec_us / profiled_batches;
        const int64_t avg_chunk_idle_us =
            total_chunk_idle_us / profiled_batches;
        const double forward_idle_ratio_total =
            total_forward_us > 0 ? static_cast<double>(total_chunk_idle_us) /
                                       static_cast<double>(total_forward_us)
                                 : 0.0;
        const double chunk_idle_nonzero_ratio_total =
            total_chunk_count > 0
                ? 100.0 * static_cast<double>(total_chunk_nonzero_count) /
                      static_cast<double>(total_chunk_count)
                : 0.0;
        const double avg_chunk_idle_nonzero_us_total =
            total_chunk_nonzero_count > 0
                ? static_cast<double>(total_chunk_idle_us) /
                      static_cast<double>(total_chunk_nonzero_count)
                : 0.0;
        const int64_t max_batch_chunk_idle_us_total =
            g_layer_exec_profiled_chunk_idle_us_max.load(
                std::memory_order_relaxed);
        const int64_t max_all_chunk_idle_us_total =
            g_layer_exec_profiled_all_chunk_idle_us_max.load(
                std::memory_order_relaxed);
        const int64_t max_batch_forward_us_total =
            g_layer_exec_profiled_forward_us_max.load(
                std::memory_order_relaxed);
        double p50_batch_forward_us_total = 0.0;
        bool has_p50_batch_forward_us_total = false;
        double p75_batch_forward_us_total = 0.0;
        bool has_p75_batch_forward_us_total = false;
        double p99_batch_forward_us_total = 0.0;
        bool has_p99_batch_forward_us_total = false;
        {
          std::lock_guard<std::mutex> lock(
              g_layer_exec_profiled_batch_forward_samples_mutex);
          if (!g_layer_exec_profiled_batch_forward_samples.empty()) {
            std::vector<int64_t> sorted_samples =
                g_layer_exec_profiled_batch_forward_samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            has_p50_batch_forward_us_total = true;
            has_p75_batch_forward_us_total = true;
            has_p99_batch_forward_us_total = true;
            p50_batch_forward_us_total =
                get_sorted_quantile_us(sorted_samples, 0.50);
            p75_batch_forward_us_total =
                get_sorted_quantile_us(sorted_samples, 0.75);
            p99_batch_forward_us_total =
                get_sorted_quantile_us(sorted_samples, 0.99);
          }
        }
        double p50_batch_idle_us_total = 0.0;
        bool has_p50_batch_idle_us_total = false;
        double p75_batch_idle_us_total = 0.0;
        bool has_p75_batch_idle_us_total = false;
        double p99_batch_idle_us_total = 0.0;
        bool has_p99_batch_idle_us_total = false;
        {
          std::lock_guard<std::mutex> lock(
              g_layer_exec_profiled_batch_idle_median_mutex);
          if (!g_layer_exec_profiled_batch_idle_lower.empty()) {
            has_p50_batch_idle_us_total = true;
            if (g_layer_exec_profiled_batch_idle_lower.size() ==
                g_layer_exec_profiled_batch_idle_upper.size()) {
              p50_batch_idle_us_total =
                  (static_cast<double>(
                       g_layer_exec_profiled_batch_idle_lower.top()) +
                   static_cast<double>(
                       g_layer_exec_profiled_batch_idle_upper.top())) /
                  2.0;
            } else {
              p50_batch_idle_us_total = static_cast<double>(
                  g_layer_exec_profiled_batch_idle_lower.top());
            }
          }
        }
        {
          std::lock_guard<std::mutex> lock(
              g_layer_exec_profiled_batch_idle_samples_mutex);
          if (!g_layer_exec_profiled_batch_idle_samples.empty()) {
            std::vector<int64_t> sorted_samples =
                g_layer_exec_profiled_batch_idle_samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            has_p75_batch_idle_us_total = true;
            has_p99_batch_idle_us_total = true;
            p75_batch_idle_us_total =
                get_sorted_quantile_us(sorted_samples, 0.75);
            p99_batch_idle_us_total =
                get_sorted_quantile_us(sorted_samples, 0.99);
          }
        }
        double p50_all_chunk_idle_us_total = 0.0;
        bool has_p50_all_chunk_idle_us_total = false;
        double p75_all_chunk_idle_us_total = 0.0;
        bool has_p75_all_chunk_idle_us_total = false;
        double p99_all_chunk_idle_us_total = 0.0;
        bool has_p99_all_chunk_idle_us_total = false;
        {
          std::lock_guard<std::mutex> lock(
              g_layer_exec_profiled_all_chunk_idle_median_mutex);
          if (!g_layer_exec_profiled_all_chunk_idle_lower.empty()) {
            has_p50_all_chunk_idle_us_total = true;
            if (g_layer_exec_profiled_all_chunk_idle_lower.size() ==
                g_layer_exec_profiled_all_chunk_idle_upper.size()) {
              p50_all_chunk_idle_us_total =
                  (static_cast<double>(
                       g_layer_exec_profiled_all_chunk_idle_lower.top()) +
                   static_cast<double>(
                       g_layer_exec_profiled_all_chunk_idle_upper.top())) /
                  2.0;
            } else {
              p50_all_chunk_idle_us_total = static_cast<double>(
                  g_layer_exec_profiled_all_chunk_idle_lower.top());
            }
          }
        }
        {
          std::lock_guard<std::mutex> lock(
              g_layer_exec_profiled_all_chunk_idle_samples_mutex);
          if (!g_layer_exec_profiled_all_chunk_idle_samples.empty()) {
            std::vector<int64_t> sorted_samples =
                g_layer_exec_profiled_all_chunk_idle_samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            has_p75_all_chunk_idle_us_total = true;
            has_p99_all_chunk_idle_us_total = true;
            p75_all_chunk_idle_us_total =
                get_sorted_quantile_us(sorted_samples, 0.75);
            p99_all_chunk_idle_us_total =
                get_sorted_quantile_us(sorted_samples, 0.99);
          }
        }
        std::vector<int64_t> chunk_idle_us_sum_total_list;
        std::vector<int64_t> chunk_sample_count_total_list;
        std::vector<int64_t> chunk_idle_us_max_total_list;
        std::vector<double> chunk_idle_us_p50_total_list;
        {
          std::lock_guard<std::mutex> lock(
              g_layer_exec_profiled_chunk_totals_mutex);
          chunk_idle_us_sum_total_list =
              g_layer_exec_profiled_chunk_idle_us_sum_list;
          chunk_sample_count_total_list =
              g_layer_exec_profiled_chunk_sample_count_list;
          chunk_idle_us_max_total_list =
              g_layer_exec_profiled_chunk_idle_us_max_list;
          chunk_idle_us_p50_total_list.resize(
              g_layer_exec_profiled_chunk_idle_lower_heaps.size(), 0.0);
          for (size_t chunk_idx = 0;
               chunk_idx < g_layer_exec_profiled_chunk_idle_lower_heaps.size();
               ++chunk_idx) {
            const auto& lower_heap =
                g_layer_exec_profiled_chunk_idle_lower_heaps[chunk_idx];
            const auto& upper_heap =
                g_layer_exec_profiled_chunk_idle_upper_heaps[chunk_idx];
            if (lower_heap.empty()) {
              chunk_idle_us_p50_total_list[chunk_idx] = 0.0;
            } else if (lower_heap.size() == upper_heap.size()) {
              chunk_idle_us_p50_total_list[chunk_idx] =
                  (static_cast<double>(lower_heap.top()) +
                   static_cast<double>(upper_heap.top())) /
                  2.0;
            } else {
              chunk_idle_us_p50_total_list[chunk_idx] =
                  static_cast<double>(lower_heap.top());
            }
          }
        }
        auto format_us_list_ms = [](const std::vector<int64_t>& values) {
          std::ostringstream oss;
          oss << "[";
          for (size_t idx = 0; idx < values.size(); ++idx) {
            if (idx > 0) {
              oss << "|";
            }
            oss << std::fixed << std::setprecision(3)
                << static_cast<double>(values[idx]) / 1000.0;
          }
          oss << "]";
          return oss.str();
        };
        auto format_int_list = [](const std::vector<int64_t>& values) {
          std::ostringstream oss;
          oss << "[";
          for (size_t idx = 0; idx < values.size(); ++idx) {
            if (idx > 0) {
              oss << "|";
            }
            oss << values[idx];
          }
          oss << "]";
          return oss.str();
        };
        auto format_double_us_list_ms = [](const std::vector<double>& values) {
          std::ostringstream oss;
          oss << "[";
          for (size_t idx = 0; idx < values.size(); ++idx) {
            if (idx > 0) {
              oss << "|";
            }
            oss << std::fixed << std::setprecision(3) << values[idx] / 1000.0;
          }
          oss << "]";
          return oss.str();
        };
        LOG(INFO)
            << "[layer_exec_profile] batches=" << profiled_batches
            << ", batch_id=" << input_params.batch_id << ", mode="
            << (input_params.batch_forward_type.is_decode() ? "decode"
                                                            : "prefill")
            << ", layers=" << layers_.size()
            << ", batch_exec_ms=" << static_cast<double>(batch_exec_us) / 1000.0
            << ", batch_interval_ms="
            << static_cast<double>(batch_interval_us) / 1000.0
            << ", batch_forward_ms="
            << static_cast<double>(batch_forward_us) / 1000.0
            << ", avg_exec_ms=" << static_cast<double>(avg_exec_us) / 1000.0
            << ", avg_interval_ms="
            << static_cast<double>(avg_interval_us) / 1000.0
            << ", avg_forward_ms="
            << static_cast<double>(avg_forward_us) / 1000.0
            << ", sum_batch_forward_ms_total="
            << static_cast<double>(total_forward_us) / 1000.0
            << ", avg_batch_forward_ms_total="
            << static_cast<double>(avg_forward_us) / 1000.0
            << ", max_batch_forward_ms_total="
            << static_cast<double>(max_batch_forward_us_total) / 1000.0
            << ", p50_batch_forward_ms_total="
            << (has_p50_batch_forward_us_total
                    ? p50_batch_forward_us_total / 1000.0
                    : 0.0)
            << ", p75_batch_forward_ms_total="
            << (has_p75_batch_forward_us_total
                    ? p75_batch_forward_us_total / 1000.0
                    : 0.0)
            << ", p99_batch_forward_ms_total="
            << (has_p99_batch_forward_us_total
                    ? p99_batch_forward_us_total / 1000.0
                    : 0.0)
            << ", forward_idle_ratio_total=" << forward_idle_ratio_total
            << ", max_layer_exec_ms(layer=" << max_layer_exec_idx
            << ")=" << static_cast<double>(max_layer_exec_us) / 1000.0
            << ", max_layer_interval_ms(before_layer=" << max_layer_interval_idx
            << ")=" << static_cast<double>(max_layer_interval_us) / 1000.0
            << ", chunk_size=" << chunk_layers
            << ", chunks=" << chunk_exec_us_list.size()
            << ", batch_chunk_exec_ms="
            << static_cast<double>(batch_chunk_exec_us) / 1000.0
            << ", batch_chunk_idle_ms="
            << static_cast<double>(batch_chunk_idle_us) / 1000.0
            << ", batch_idle_ms="
            << static_cast<double>(batch_chunk_idle_us) / 1000.0
            << ", avg_chunk_exec_ms="
            << static_cast<double>(avg_chunk_exec_us) / 1000.0
            << ", avg_chunk_idle_ms="
            << static_cast<double>(avg_chunk_idle_us) / 1000.0
            << ", sum_batch_idle_ms_total="
            << static_cast<double>(total_chunk_idle_us) / 1000.0
            << ", avg_batch_idle_ms_total="
            << static_cast<double>(avg_chunk_idle_us) / 1000.0
            << ", sum_chunk_idle_ms_total="
            << static_cast<double>(total_chunk_idle_us) / 1000.0
            << ", total_chunk_count=" << total_chunk_count
            << ", total_chunk_nonzero_count=" << total_chunk_nonzero_count
            << ", chunk_idle_nonzero_ratio_total(%)="
            << chunk_idle_nonzero_ratio_total
            << ", avg_chunk_idle_ms_when_nonzero_total="
            << avg_chunk_idle_nonzero_us_total / 1000.0
            // Historical name was max_chunk_idle_ms_total; this is
            // actually max of batch_chunk_idle_ms (sum of chunk idles).
            << ", max_batch_idle_ms_sum_chunks_total="
            << static_cast<double>(max_batch_chunk_idle_us_total) / 1000.0
            << ", p50_batch_idle_ms_sum_chunks_total="
            << (has_p50_batch_idle_us_total ? p50_batch_idle_us_total / 1000.0
                                            : 0.0)
            << ", p75_batch_idle_ms_sum_chunks_total="
            << (has_p75_batch_idle_us_total ? p75_batch_idle_us_total / 1000.0
                                            : 0.0)
            << ", p99_batch_idle_ms_sum_chunks_total="
            << (has_p99_batch_idle_us_total ? p99_batch_idle_us_total / 1000.0
                                            : 0.0)
            << ", p50_all_chunks_idle_ms_total="
            << (has_p50_all_chunk_idle_us_total
                    ? p50_all_chunk_idle_us_total / 1000.0
                    : 0.0)
            << ", p75_all_chunks_idle_ms_total="
            << (has_p75_all_chunk_idle_us_total
                    ? p75_all_chunk_idle_us_total / 1000.0
                    : 0.0)
            << ", p99_all_chunks_idle_ms_total="
            << (has_p99_all_chunk_idle_us_total
                    ? p99_all_chunk_idle_us_total / 1000.0
                    : 0.0)
            << ", max_all_chunks_idle_ms_total="
            << static_cast<double>(max_all_chunk_idle_us_total) / 1000.0
            << ", max_chunk_exec_ms(chunk=" << max_chunk_exec_idx
            << ")=" << static_cast<double>(max_chunk_exec_us) / 1000.0
            << ", max_chunk_idle_ms(before_chunk=" << max_chunk_idle_idx
            << ")=" << static_cast<double>(max_chunk_idle_us) / 1000.0
            << ", chunk_idle_ms_sum_total_list="
            << format_us_list_ms(chunk_idle_us_sum_total_list)
            << ", chunk_sample_count_total_list="
            << format_int_list(chunk_sample_count_total_list)
            << ", chunk_idle_ms_max_total_list="
            << format_us_list_ms(chunk_idle_us_max_total_list)
            << ", chunk_idle_ms_p50_total_list="
            << format_double_us_list_ms(chunk_idle_us_p50_total_list)
            << ", layer_exec_ms_list=" << format_us_list_ms(layer_exec_us_list)
            << ", layer_interval_ms_before_layer_list="
            << format_us_list_ms(layer_interval_us_list)
            << ", chunk_exec_ms_list=" << format_us_list_ms(chunk_exec_us_list)
            << ", chunk_idle_ms_before_chunk_list="
            << format_us_list_ms(chunk_idle_us_list);
      }
    }

    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    npu_embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    npu_embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");

    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  virtual void merge_loaded_weights() {
    npu_embed_tokens_->merge_loaded_weights();

    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  virtual void free_weights() {
    npu_embed_tokens_->free_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->free_weights();
    }
    norm_->free_weights();
  }

  virtual void reload_weights() {
    npu_embed_tokens_->reload_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights();
    }
    norm_->reload_weights();
  }

  virtual void reload_non_decoder_weights() {
    npu_embed_tokens_->reload_weights();
    norm_->reload_weights();
  }

  virtual void reload_weights_from_device() {
    npu_embed_tokens_->reload_weights_from_device();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights_from_device();
    }
    norm_->reload_weights_from_device();
  }

  virtual void merge_and_move_pinned_host() {
    npu_embed_tokens_->merge_and_move_pinned_host();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_and_move_pinned_host();
    }
    norm_->merge_and_move_pinned_host();
  }

  // Collect BaseManualLoader* from each decoder layer (in order)
  virtual std::vector<layer::BaseManualLoader*> get_decoder_loaders() {
    std::vector<layer::BaseManualLoader*> loaders;
    loaders.reserve(layers_.size());
    for (auto& l : layers_) {
      loaders.push_back(l->get_manual_loader());
    }
    return loaders;
  }

  // Inject rolling load manager (not owned, managed by WorkerImpl)
  void set_rolling_load_manager(RollingLoadManager* mgr) { rolling_mgr_ = mgr; }

  // For rolling load: refresh decoder layers' rolling device pointers and
  // corresponding AT/ATB tensor bindings.
  virtual void refresh_rolling_weights() {
    for (auto& layer : layers_) {
      layer->refresh_rolling_weights();
    }
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return npu_embed_tokens_;
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }

 protected:
  torch::Tensor cos_sin_;
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  int device_id = 0;
  layer::AttentionMask attn_mask_;
  int dp_rank_ = 0;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};

  std::vector<int64_t> mrope_section_;
  // test
  //  ParallelEmbedding embed_tokens_{nullptr};
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  layer::NpuRMSNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayerType> layers_;

  bool layer_forward_interrupted_ = false;

  int32_t max_seq_len_ = 0;

  RollingLoadManager* rolling_mgr_ =
      nullptr;  // not owned; managed by LlmForCausalLMImplBase

 private:
  std::string model_type_;
};

template <typename LlmModelType>
class LlmForCausalLMImplBase : public torch::nn::Module {
 public:
  LlmForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    // register submodules
    model_ = register_module("model", LlmModelType(context));

    npu_lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    return npu_lm_head_(hidden_states, seleted_idxes, 0);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  virtual void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);

      if (tie_word_embeddings) {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("lm_head."));
      }
    }

    // verify
    model_->verify_loaded_weights(prefix);
    if (tie_word_embeddings) {
      npu_lm_head_->verify_loaded_weights(prefix + "embed_tokens.");
    } else {
      npu_lm_head_->verify_loaded_weights("lm_head.");
    }

    model_->merge_loaded_weights();
    // test
    npu_lm_head_->merge_loaded_weights();
  }

  virtual void lazy_load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) {
    if (keep_host_weights) {
      LOG(INFO) << "Model weights are already kept on host.";
      return;
    }
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings) {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("lm_head."));
      }
    }
    // verify
    model_->verify_loaded_weights(prefix);
    npu_lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_and_move_pinned_host();
    // test
    npu_lm_head_->merge_and_move_pinned_host();

    keep_host_weights = true;
  }

  virtual void free_model_weights() {
    if (!keep_host_weights) {
      LOG(INFO) << "Model weights are not kept on host.";
      return;
    }
    model_->free_weights();
    npu_lm_head_->free_weights();
    keep_host_weights = false;
  }

  virtual void reload_model_weights() {
    model_->reload_weights();
    npu_lm_head_->reload_weights();
    auto stream = c10_npu::getCurrentNPUStream();
    stream.synchronize();
  }

  virtual void init_rolling_model_state() {
    model_->reload_non_decoder_weights();
    model_->refresh_rolling_weights();
    npu_lm_head_->reload_weights();
    auto stream = c10_npu::getCurrentNPUStream();
    stream.synchronize();
  }

  virtual void reload_model_weights_from_device() {
    model_->reload_weights_from_device();
    npu_lm_head_->reload_weights_from_device();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::NpuLmHead get_npu_lm_head() { return npu_lm_head_; }

  virtual void set_npu_lm_head(layer::NpuLmHead& head) { npu_lm_head_ = head; }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

  virtual std::vector<layer::BaseManualLoader*> get_decoder_loaders() {
    return model_->get_decoder_loaders();
  }

  virtual void set_rolling_load_manager(RollingLoadManager* mgr) {
    model_->set_rolling_load_manager(mgr);
  }

  virtual bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                               Stream* compute_stream,
                                               int32_t num_cached_slots,
                                               int32_t requested_rolling_slots,
                                               const std::string& model_id) {
    CHECK(load_stream != nullptr) << "load_stream is null for rolling load";
    CHECK(compute_stream != nullptr)
        << "compute_stream is null for rolling load";

    if (rolling_load_manager_ == nullptr) {
      auto loaders = model_->get_decoder_loaders();
      CHECK(!loaders.empty()) << "No decoder loaders found for rolling load";
      size_t max_storage_size = 0;
      for (size_t i = 0; i < loaders.size(); ++i) {
        CHECK(loaders[i] != nullptr) << "Decoder loader[" << i << "] is null";
        const size_t layer_storage_size = loaders[i]->get_storage_size();
        CHECK_GT(layer_storage_size, 0)
            << "Decoder loader[" << i << "] invalid storage_size";
        if (layer_storage_size > max_storage_size) {
          max_storage_size = layer_storage_size;
        }
      }
      CHECK_GT(max_storage_size, 0)
          << "Failed to determine max decoder layer storage_size";

      rolling_weight_buffer_ = std::make_shared<layer::RollingWeightBuffer>(
          num_cached_slots, max_storage_size, model_id);
      rolling_load_manager_ =
          std::make_unique<RollingLoadManager>(loaders,
                                               rolling_weight_buffer_,
                                               load_stream,
                                               compute_stream,
                                               requested_rolling_slots);
      LOG(INFO) << "Rolling runtime init: num_cached_slots=" << num_cached_slots
                << ", max_decoder_layer_storage_size=" << max_storage_size;

      for (size_t i = 0; i < loaders.size(); ++i) {
        const int32_t layer_index = i;
        const int32_t slot = rolling_load_manager_->slot_for_layer(layer_index);
        loaders[i]->set_rolling_buffer(rolling_weight_buffer_, slot);
      }
    } else {
      rolling_load_manager_->refresh_rolling_buffer_address();
    }

    model_->set_rolling_load_manager(rolling_load_manager_.get());
    init_rolling_model_state();
    rolling_load_manager_->init_rolling_load();
    return true;
  }

 protected:
  // parameter members, must be registered
  LlmModelType model_{nullptr};
  int device_id = 0;
  bool tie_word_embeddings{false};
  bool keep_host_weights{false};
  std::shared_ptr<layer::RollingWeightBuffer> rolling_weight_buffer_{nullptr};
  std::unique_ptr<RollingLoadManager> rolling_load_manager_{nullptr};
  // test
  layer::NpuLmHead npu_lm_head_{nullptr};
};

}  // namespace xllm
