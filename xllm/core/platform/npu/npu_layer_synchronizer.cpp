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
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/global_flags.h"

namespace xllm {

namespace {
std::atomic<int64_t> g_profiled_batches{0};
std::atomic<int64_t> g_profiled_copy_us{0};
std::atomic<int64_t> g_profiled_copy_blocks{0};
std::atomic<int64_t> g_profiled_window_batches{0};
std::atomic<int64_t> g_profiled_window_copy_us{0};
std::atomic<int64_t> g_profiled_window_copy_blocks{0};
double g_profiled_batch_t_per_sum_ms_per_block = 0.0;
std::mutex g_profiled_batch_t_per_sum_ms_mutex;

std::mutex g_t_per_ms_per_block_mutex;
std::vector<double> g_t_per_ms_per_block_samples;

struct LayerChunkTotals {
  int64_t total_copy_us = 0;
  int64_t total_copy_blocks = 0;
  int64_t start_layer = -1;
  int64_t end_layer = -1;

  int64_t window_copy_us = 0;
  int64_t window_copy_blocks = 0;
};

std::mutex g_layer_chunk_totals_mutex;
std::unordered_map<int64_t, LayerChunkTotals> g_layer_chunk_totals;

double get_sorted_quantile(const std::vector<double>& sorted_values, double q) {
  if (sorted_values.empty()) {
    return -1.0;
  }
  const double clamped_q = std::max(0.0, std::min(1.0, q));
  const double pos = (sorted_values.size() - 1) * clamped_q;
  const size_t low_idx = static_cast<size_t>(std::floor(pos));
  const size_t high_idx = static_cast<size_t>(std::ceil(pos));
  if (low_idx == high_idx) {
    return sorted_values[low_idx];
  }
  const double low_v = sorted_values[low_idx];
  const double high_v = sorted_values[high_idx];
  const double frac = pos - low_idx;
  return low_v + frac * (high_v - low_v);
}
}  // namespace

NPULayerSynchronizerImpl::NPULayerSynchronizerImpl(const int64_t num_layers,
                                                   const int32_t timeout)
    : events_(num_layers, nullptr),
      event_record_flags_(num_layers),
      h2d_copy_time_us_per_layer_(num_layers),
      h2d_copy_blocks_per_layer_(num_layers),
      layer_range_start_(num_layers, -1),
      layer_range_end_(num_layers, -1),
      timeout_(timeout) {
  uint32_t flags = ACL_EVENT_SYNC;
  for (int64_t i = 0; i < num_layers; ++i) {
    h2d_copy_time_us_per_layer_[i].store(0, std::memory_order_relaxed);
    h2d_copy_blocks_per_layer_[i].store(0, std::memory_order_relaxed);
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK(ret == ACL_SUCCESS) << "Create event failed:" << ret;
  }
}

NPULayerSynchronizerImpl::~NPULayerSynchronizerImpl() {
  if (FLAGS_enable_h2d_overlap_profile) {
    const int64_t batch_copy_us =
        h2d_copy_time_us_.load(std::memory_order_relaxed);
    const int64_t batch_copy_blocks =
        h2d_copy_blocks_.load(std::memory_order_relaxed);
    if (batch_copy_us > 0 && batch_copy_blocks > 0) {
      const int64_t batch_idx =
          g_profiled_batches.fetch_add(1, std::memory_order_relaxed) + 1;
      const int64_t total_copy_us =
          g_profiled_copy_us.fetch_add(batch_copy_us,
                                       std::memory_order_relaxed) +
          batch_copy_us;
      const int64_t total_copy_blocks =
          g_profiled_copy_blocks.fetch_add(batch_copy_blocks,
                                           std::memory_order_relaxed) +
          batch_copy_blocks;
      const int64_t window_batches =
          g_profiled_window_batches.fetch_add(1, std::memory_order_relaxed) + 1;
      const int64_t window_copy_us =
          g_profiled_window_copy_us.fetch_add(batch_copy_us,
                                              std::memory_order_relaxed) +
          batch_copy_us;
      const int64_t window_copy_blocks =
          g_profiled_window_copy_blocks.fetch_add(batch_copy_blocks,
                                                  std::memory_order_relaxed) +
          batch_copy_blocks;
      const int64_t interval =
          std::max<int64_t>(1, FLAGS_h2d_overlap_profile_log_interval);
      const bool should_log = (batch_idx % interval == 0);
      const double batch_t_per_ms_per_block =
          static_cast<double>(batch_copy_us) / batch_copy_blocks / 1000.0;
      const double total_t_per_ms_per_block =
          static_cast<double>(total_copy_us) / total_copy_blocks / 1000.0;
      const double window_t_per_ms_per_block =
          static_cast<double>(window_copy_us) / window_copy_blocks / 1000.0;
      double total_t_per_p25_ms_per_block = -1.0;
      double total_t_per_p50_ms_per_block = -1.0;
      double total_t_per_p75_ms_per_block = -1.0;
      double total_t_per_p99_ms_per_block = -1.0;
      double total_batch_t_per_avg_ms_per_block = -1.0;
      {
        std::lock_guard<std::mutex> lock(g_t_per_ms_per_block_mutex);
        g_t_per_ms_per_block_samples.push_back(batch_t_per_ms_per_block);
        if (should_log) {
          std::vector<double> sorted_samples = g_t_per_ms_per_block_samples;
          std::sort(sorted_samples.begin(), sorted_samples.end());
          total_t_per_p25_ms_per_block =
              get_sorted_quantile(sorted_samples, 0.25);
          total_t_per_p50_ms_per_block =
              get_sorted_quantile(sorted_samples, 0.50);
          total_t_per_p75_ms_per_block =
              get_sorted_quantile(sorted_samples, 0.75);
          total_t_per_p99_ms_per_block =
              get_sorted_quantile(sorted_samples, 0.99);
        }
      }
      {
        std::lock_guard<std::mutex> lock(g_profiled_batch_t_per_sum_ms_mutex);
        g_profiled_batch_t_per_sum_ms_per_block += batch_t_per_ms_per_block;
        if (should_log && batch_idx > 0) {
          total_batch_t_per_avg_ms_per_block =
              g_profiled_batch_t_per_sum_ms_per_block /
              static_cast<double>(batch_idx);
        }
      }

      struct LayerLogSnapshot {
        int64_t layer_idx = -1;
        int64_t start_layer = -1;
        int64_t end_layer = -1;
        int64_t batch_copy_us = 0;
        int64_t batch_copy_blocks = 0;
        int64_t total_copy_us = 0;
        int64_t total_copy_blocks = 0;
        int64_t window_copy_us = 0;
        int64_t window_copy_blocks = 0;
      };
      std::vector<LayerLogSnapshot> layer_logs;
      const bool reset_window = should_log && (window_batches >= interval);
      for (int64_t layer_idx = 0; layer_idx < events_.size(); ++layer_idx) {
        const int64_t layer_copy_us =
            h2d_copy_time_us_per_layer_[layer_idx].load(
                std::memory_order_relaxed);
        const int64_t layer_copy_blocks =
            h2d_copy_blocks_per_layer_[layer_idx].load(
                std::memory_order_relaxed);
        if (layer_copy_us <= 0 || layer_copy_blocks <= 0) {
          continue;
        }
        const int64_t start_layer = layer_range_start_[layer_idx];
        const int64_t end_layer = layer_range_end_[layer_idx];

        LayerLogSnapshot snapshot;
        snapshot.layer_idx = layer_idx;
        snapshot.start_layer = start_layer;
        snapshot.end_layer = end_layer;
        snapshot.batch_copy_us = layer_copy_us;
        snapshot.batch_copy_blocks = layer_copy_blocks;
        {
          std::lock_guard<std::mutex> lock(g_layer_chunk_totals_mutex);
          auto& totals = g_layer_chunk_totals[layer_idx];
          totals.total_copy_us += layer_copy_us;
          totals.total_copy_blocks += layer_copy_blocks;
          totals.window_copy_us += layer_copy_us;
          totals.window_copy_blocks += layer_copy_blocks;
          if (start_layer >= 0 && end_layer >= 0) {
            totals.start_layer = start_layer;
            totals.end_layer = end_layer;
          }
          if (should_log) {
            snapshot.total_copy_us = totals.total_copy_us;
            snapshot.total_copy_blocks = totals.total_copy_blocks;
            snapshot.window_copy_us = totals.window_copy_us;
            snapshot.window_copy_blocks = totals.window_copy_blocks;
          }
          if (reset_window) {
            totals.window_copy_us = 0;
            totals.window_copy_blocks = 0;
          }
        }
        if (should_log) {
          layer_logs.push_back(snapshot);
        }
      }

      if (should_log) {
        LOG(INFO)
            << "[h2d_overlap_profile] batches=" << batch_idx
            << ", batch_copy_ms=" << batch_copy_us / 1000.0
            << ", batch_copy_blocks=" << batch_copy_blocks
            << ", batch_t_per_ms_per_block=" << batch_t_per_ms_per_block
            << ", total_copy_ms=" << total_copy_us / 1000.0
            << ", total_copy_blocks=" << total_copy_blocks
            << ", total_t_per_ms_per_block=" << total_t_per_ms_per_block
            << ", window_copy_ms=" << window_copy_us / 1000.0
            << ", window_copy_blocks=" << window_copy_blocks
            << ", window_t_per_ms_per_block=" << window_t_per_ms_per_block
            << ", total_t_per_p25_ms_per_block=" << total_t_per_p25_ms_per_block
            << ", total_t_per_p50_ms_per_block=" << total_t_per_p50_ms_per_block
            << ", total_t_per_p75_ms_per_block=" << total_t_per_p75_ms_per_block
            << ", total_t_per_p99_ms_per_block=" << total_t_per_p99_ms_per_block
            << ", total_batch_t_per_avg_ms_per_block="
            << total_batch_t_per_avg_ms_per_block;
        for (const auto& layer_log : layer_logs) {
          LOG(INFO) << "[h2d_overlap_profile][layer] batches=" << batch_idx
                    << ", layer_chunk_idx=" << layer_log.layer_idx
                    << ", layer_range=[" << layer_log.start_layer << ","
                    << layer_log.end_layer << ")"
                    << ", batch_copy_ms=" << layer_log.batch_copy_us / 1000.0
                    << ", batch_copy_blocks=" << layer_log.batch_copy_blocks
                    << ", batch_t_per_ms_per_block="
                    << static_cast<double>(layer_log.batch_copy_us) /
                           layer_log.batch_copy_blocks / 1000.0
                    << ", total_copy_ms=" << layer_log.total_copy_us / 1000.0
                    << ", total_copy_blocks=" << layer_log.total_copy_blocks
                    << ", total_t_per_ms_per_block="
                    << static_cast<double>(layer_log.total_copy_us) /
                           layer_log.total_copy_blocks / 1000.0
                    << ", window_copy_ms=" << layer_log.window_copy_us / 1000.0
                    << ", window_copy_blocks=" << layer_log.window_copy_blocks
                    << ", window_t_per_ms_per_block="
                    << static_cast<double>(layer_log.window_copy_us) /
                           layer_log.window_copy_blocks / 1000.0;
        }
        if (reset_window) {
          g_profiled_window_batches.store(0, std::memory_order_relaxed);
          g_profiled_window_copy_us.store(0, std::memory_order_relaxed);
          g_profiled_window_copy_blocks.store(0, std::memory_order_relaxed);
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
                                                    int64_t elapsed_us,
                                                    int64_t copy_blocks) {
  h2d_copy_time_us_per_layer_[layer_index].fetch_add(elapsed_us,
                                                     std::memory_order_relaxed);
  h2d_copy_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
  if (copy_blocks > 0) {
    h2d_copy_blocks_per_layer_[layer_index].fetch_add(
        copy_blocks, std::memory_order_relaxed);
    int64_t prev_copy_blocks = h2d_copy_blocks_.load(std::memory_order_relaxed);
    while (prev_copy_blocks < copy_blocks &&
           !h2d_copy_blocks_.compare_exchange_weak(prev_copy_blocks,
                                                   copy_blocks,
                                                   std::memory_order_relaxed,
                                                   std::memory_order_relaxed)) {
    }
  }
}

void NPULayerSynchronizerImpl::set_layer_range(int64_t layer_index,
                                               int64_t start_layer,
                                               int64_t end_layer) {
  layer_range_start_[layer_index] = start_layer;
  layer_range_end_[layer_index] = end_layer;
}

}  // namespace xllm
