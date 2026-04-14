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

#include "npu_layer_synchronizer.h"

#include <glog/logging.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "common/global_flags.h"

namespace xllm {

namespace {
std::atomic<int64_t> g_profiled_batches{0};
std::atomic<int64_t> g_profiled_copy_us{0};

struct LayerChunkTotals {
  int64_t total_copy_us = 0;
  int64_t start_layer = -1;
  int64_t end_layer = -1;
};

std::mutex g_layer_chunk_totals_mutex;
std::unordered_map<int64_t, LayerChunkTotals> g_layer_chunk_totals;
}  // namespace

NPULayerSynchronizerImpl::NPULayerSynchronizerImpl(const int64_t num_layers,
                                                   const int32_t timeout)
    : events_(num_layers, nullptr),
      event_record_flags_(num_layers),
      h2d_copy_time_us_per_layer_(num_layers),
      layer_range_start_(num_layers, -1),
      layer_range_end_(num_layers, -1),
      timeout_(timeout) {
  uint32_t flags = ACL_EVENT_SYNC;
  for (int64_t i = 0; i < num_layers; ++i) {
    h2d_copy_time_us_per_layer_[i].store(0, std::memory_order_relaxed);
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK(ret == ACL_SUCCESS) << "Create event failed:" << ret;
  }
}

NPULayerSynchronizerImpl::~NPULayerSynchronizerImpl() {
  if (FLAGS_enable_h2d_overlap_profile) {
    const int64_t batch_copy_us =
        h2d_copy_time_us_.load(std::memory_order_relaxed);
    if (batch_copy_us > 0) {
      const int64_t batch_idx =
          g_profiled_batches.fetch_add(1, std::memory_order_relaxed) + 1;
      const int64_t total_copy_us =
          g_profiled_copy_us.fetch_add(batch_copy_us,
                                       std::memory_order_relaxed) +
          batch_copy_us;
      const int64_t interval =
          std::max<int64_t>(1, FLAGS_h2d_overlap_profile_log_interval);
      if (batch_idx % interval == 0) {
        LOG(INFO) << "[h2d_overlap_profile] batches=" << batch_idx
                  << ", batch_copy_ms=" << batch_copy_us / 1000.0
                  << ", total_copy_ms=" << total_copy_us / 1000.0;

        for (int64_t layer_idx = 0; layer_idx < events_.size(); ++layer_idx) {
          const int64_t layer_copy_us =
              h2d_copy_time_us_per_layer_[layer_idx].load(
                  std::memory_order_relaxed);
          if (layer_copy_us <= 0) {
            continue;
          }
          const int64_t start_layer = layer_range_start_[layer_idx];
          const int64_t end_layer = layer_range_end_[layer_idx];

          int64_t layer_total_copy_us = 0;
          {
            std::lock_guard<std::mutex> lock(g_layer_chunk_totals_mutex);
            auto& totals = g_layer_chunk_totals[layer_idx];
            totals.total_copy_us += layer_copy_us;
            if (start_layer >= 0 && end_layer >= 0) {
              totals.start_layer = start_layer;
              totals.end_layer = end_layer;
            }
            layer_total_copy_us = totals.total_copy_us;
          }

          LOG(INFO) << "[h2d_overlap_profile][layer] batches=" << batch_idx
                    << ", layer_chunk_idx=" << layer_idx << ", layer_range=["
                    << start_layer << "," << end_layer << ")"
                    << ", batch_copy_ms=" << layer_copy_us / 1000.0
                    << ", total_copy_ms=" << layer_total_copy_us / 1000.0;
        }
      }
    }
  }
  for (int64_t i = 0; i < events_.size(); ++i) {
    aclrtDestroyEvent(events_[i]);
  }
}

aclrtEvent* NPULayerSynchronizerImpl::get_event(const int64_t layer_index) {
  return &events_[layer_index];
}

std::atomic<bool>* NPULayerSynchronizerImpl::get_event_flag(
    const int64_t layer_index) {
  return &event_record_flags_[layer_index];
}

bool NPULayerSynchronizerImpl::synchronize_layer(const int64_t layer_index) {
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire));
  const aclError ret =
      aclrtSynchronizeEventWithTimeout(events_[layer_index], timeout_);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Synchronize event failed: " << ret;
    return false;
  }
  return true;
}

void NPULayerSynchronizerImpl::add_h2d_copy_time_us(int64_t layer_index,
                                                    int64_t elapsed_us) {
  h2d_copy_time_us_per_layer_[layer_index].fetch_add(elapsed_us,
                                                     std::memory_order_relaxed);
  h2d_copy_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
}

void NPULayerSynchronizerImpl::set_layer_range(int64_t layer_index,
                                               int64_t start_layer,
                                               int64_t end_layer) {
  layer_range_start_[layer_index] = start_layer;
  layer_range_end_[layer_index] = end_layer;
}

}  // namespace xllm
