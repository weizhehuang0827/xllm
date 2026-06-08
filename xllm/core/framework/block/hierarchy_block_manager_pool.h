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

#include <limits>

#include "block_manager_pool.h"
#include "distributed_runtime/engine.h"
#include "util/blockingconcurrentqueue.h"

namespace xllm {

class Engine;

struct OffloadBlockPair {
  OffloadBlockPair(Block& s,
                   Block& d,
                   bool release_blocks_after_transfer_ = true)
      : src(s),
        dst(d),
        release_blocks_after_transfer(release_blocks_after_transfer_) {}

  OffloadBlockPair(Block&& s,
                   Block&& d,
                   bool release_blocks_after_transfer_ = true)
      : src(std::move(s)),
        dst(std::move(d)),
        release_blocks_after_transfer(release_blocks_after_transfer_) {}

  OffloadBlockPair(Block& s, bool release_blocks_after_transfer_ = true)
      : src(s), release_blocks_after_transfer(release_blocks_after_transfer_) {}

  OffloadBlockPair(Block&& s, bool release_blocks_after_transfer_ = true)
      : src(std::move(s)),
        release_blocks_after_transfer(release_blocks_after_transfer_) {}

  Block src;
  Block dst;
  bool release_blocks_after_transfer = true;
};

class HierarchyBlockManagerPool : public BlockManagerPool {
 public:
  using OffloadBlockPairQueue =
      moodycamel::BlockingConcurrentQueue<std::shared_ptr<OffloadBlockPair>>;

  explicit HierarchyBlockManagerPool(const BlockManagerPool::Options& options,
                                     Engine* engine,
                                     int32_t dp_size = 1);
  ~HierarchyBlockManagerPool() = default;

  bool allocate(Sequence* sequence, size_t num_tokens) override;

  // control the copy in blocks num
  bool allocate(Sequence* sequence,
                size_t num_tokens,
                size_t max_copy_in_blocks_num) override;

  void allocate_shared(Sequence* sequence) override;

  void deallocate(Sequence* sequence) override;
  size_t enqueue_running_d2h_blocks(
      Sequence* sequence,
      size_t max_offload_blocks = std::numeric_limits<size_t>::max()) override;

  void transfer_blocks(std::vector<Batch>& batches) override;
  void transfer_blocks() override;

  void prefetch_from_storage(std::shared_ptr<Request>& request) override;

  bool update_prefetch_result(std::shared_ptr<Request>& request,
                              const uint32_t timeout) override;

  // Profile-only helpers for constructing a sequence whose KV blocks already
  // live in host memory while device KV slots are allocated but uncached.
  bool allocate_device_blocks_for_profile(Sequence* sequence,
                                          size_t num_tokens);
  bool allocate_host_blocks_for_profile(Sequence* sequence, size_t num_tokens);
  void deallocate_host_and_device_without_cache_for_profile(Sequence* sequence);

  void get_merged_kvcache_event(KvCacheEvent* event) const override;

 private:
  void allocate_host_shared(Sequence* sequence);
  size_t transfer_offload_blocks();
  void maybe_log_transfer_profile(size_t step_h2d_blocks,
                                  size_t step_d2h_blocks);

 private:
  Engine* engine_;
  std::vector<std::unique_ptr<BlockManager>> host_block_managers_;

  // BlockTransferInfo per step
  std::vector<std::vector<BlockTransferInfo>> load_block_transfer_infos_;
  std::vector<OffloadBlockPairQueue> offload_block_pair_queues_;

  // Step-level transfer profile (windowed log).
  size_t transfer_profiled_steps_ = 0;
  size_t transfer_profile_h2d_blocks_total_ = 0;
  size_t transfer_profile_d2h_blocks_total_ = 0;
  size_t transfer_profile_h2d_blocks_window_ = 0;
  size_t transfer_profile_d2h_blocks_window_ = 0;
  size_t transfer_profile_h2d_blocks_window_max_ = 0;
  size_t transfer_profile_d2h_blocks_window_max_ = 0;
  size_t transfer_profile_h2d_blocks_total_max_ = 0;
  size_t transfer_profile_d2h_blocks_total_max_ = 0;
  size_t transfer_profile_d2h_blocks_actual_total_ = 0;
  size_t transfer_profile_d2h_blocks_actual_window_ = 0;
  size_t transfer_profile_d2h_blocks_actual_window_max_ = 0;
  size_t transfer_profile_d2h_blocks_actual_total_max_ = 0;
  size_t transfer_profile_d2h_blocks_synthetic_total_ = 0;
  size_t transfer_profile_d2h_blocks_synthetic_window_ = 0;
  size_t transfer_profile_d2h_blocks_synthetic_window_max_ = 0;
  size_t transfer_profile_d2h_blocks_synthetic_total_max_ = 0;
  size_t transfer_profile_window_steps_ = 0;
};

}  // namespace xllm
