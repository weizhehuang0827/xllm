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

#include "core/framework/speculative/speculative_profile_registry.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <utility>

namespace xllm {
namespace {

double sanitize_non_negative(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(value, 0.0);
}

}  // namespace

SpeculativeProfileRegistry& SpeculativeProfileRegistry::get_instance() {
  static SpeculativeProfileRegistry registry;
  return registry;
}

void SpeculativeProfileRegistry::set_validate_time_predictor(
    const ValidateTimePredictor& predictor) {
  ValidateTimePredictor sanitized_predictor = predictor;
  sanitized_predictor.intercept_ms =
      sanitize_non_negative(predictor.intercept_ms);
  sanitized_predictor.query_token_ms =
      sanitize_non_negative(predictor.query_token_ms);
  sanitized_predictor.query_prefix_ms =
      sanitize_non_negative(predictor.query_prefix_ms);
  std::lock_guard<std::mutex> lock(mutex_);
  validate_time_predictor_ = sanitized_predictor;
}

void SpeculativeProfileRegistry::reset_validate_time_predictor() {
  std::lock_guard<std::mutex> lock(mutex_);
  validate_time_predictor_.reset();
}

bool SpeculativeProfileRegistry::has_validate_time_predictor() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return validate_time_predictor_.has_value();
}

std::optional<SpeculativeProfileRegistry::ValidateTimePredictor>
SpeculativeProfileRegistry::validate_time_predictor() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return validate_time_predictor_;
}

void SpeculativeProfileRegistry::set_sps_cost_table(const SpsCostTable& table) {
  // Keep only probes with a finite, strictly positive steps_per_sec, and
  // enforce a strictly-increasing batch_tokens axis so the floor-probe lookup
  // is well defined. A probe whose batch_tokens does not advance past the last
  // kept one is dropped (the sweep can repeat batch_tokens across (batch,
  // query) pairs).
  SpsCostTable sanitized;
  sanitized.sample_batch_tokens.reserve(table.sample_batch_tokens.size());
  sanitized.sample_steps_per_sec.reserve(table.sample_steps_per_sec.size());
  const size_t count = std::min(table.sample_batch_tokens.size(),
                                table.sample_steps_per_sec.size());
  for (size_t i = 0; i < count; ++i) {
    const int32_t batch_tokens = table.sample_batch_tokens[i];
    const double steps_per_sec = table.sample_steps_per_sec[i];
    if (batch_tokens < 1 || !std::isfinite(steps_per_sec) ||
        steps_per_sec <= 0.0) {
      continue;
    }
    if (!sanitized.sample_batch_tokens.empty() &&
        batch_tokens <= sanitized.sample_batch_tokens.back()) {
      continue;
    }
    sanitized.sample_batch_tokens.push_back(batch_tokens);
    sanitized.sample_steps_per_sec.push_back(steps_per_sec);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (sanitized.sample_batch_tokens.empty()) {
    sps_cost_table_.reset();
    return;
  }
  sps_cost_table_ = std::move(sanitized);
}

void SpeculativeProfileRegistry::reset_sps_cost_table() {
  std::lock_guard<std::mutex> lock(mutex_);
  sps_cost_table_.reset();
}

bool SpeculativeProfileRegistry::has_sps_cost_table() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sps_cost_table_.has_value();
}

std::optional<SpeculativeProfileRegistry::SpsCostTable>
SpeculativeProfileRegistry::sps_cost_table() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sps_cost_table_;
}

}  // namespace xllm
