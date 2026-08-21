/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace xllm {

// Thread-safe singleton storing the speculative profiling results fitted by
// ProfileManager and consumed by AdaptiveSpeculativeController.
//
// It holds two cost models, both produced from the same DECODE validate sweep:
//   * ValidateTimePredictor -- a linear per-seq validate-time model. This is
//     also the gate for the adaptive path: adaptive decode only activates once
//     the predictor is set (regardless of which cost model the controller
//     picks), because it is what every worker keys pruning on.
//   * SpsCostTable -- a batch_tokens -> steps_per_sec lookup table for the
//     goodput (steps-per-second) objective. Optional: only consumed when
//     --speculative_adaptive_cost_model=sps.
class SpeculativeProfileRegistry final {
 public:
  struct ValidateTimePredictor {
    double intercept_ms = 0.0;
    double query_token_ms = 0.0;
    double query_prefix_ms = 0.0;
  };

  // Piecewise-constant steps-per-second cost table, keyed by the total number
  // of validate rows in a step (batch_tokens). sample_batch_tokens is strictly
  // increasing and parallel to sample_steps_per_sec. Lookup floor-probes the
  // batch_tokens axis (see sps_lookup in the controller). Pure data, so the
  // lookup lives outside the struct per the project style rule.
  struct SpsCostTable {
    std::vector<int32_t> sample_batch_tokens;
    std::vector<double> sample_steps_per_sec;
  };

  static SpeculativeProfileRegistry& get_instance();

  void set_validate_time_predictor(const ValidateTimePredictor& predictor);
  void reset_validate_time_predictor();

  bool has_validate_time_predictor() const;
  std::optional<ValidateTimePredictor> validate_time_predictor() const;

  // Sanitizes (drops non-finite / non-positive steps_per_sec and their probe,
  // then requires strictly-increasing batch_tokens) before committing. A table
  // with no usable probe left is dropped, leaving has_sps_cost_table() false.
  void set_sps_cost_table(const SpsCostTable& table);
  void reset_sps_cost_table();

  bool has_sps_cost_table() const;
  std::optional<SpsCostTable> sps_cost_table() const;

 private:
  SpeculativeProfileRegistry() = default;

  mutable std::mutex mutex_;
  std::optional<ValidateTimePredictor> validate_time_predictor_;
  std::optional<SpsCostTable> sps_cost_table_;
};

}  // namespace xllm
