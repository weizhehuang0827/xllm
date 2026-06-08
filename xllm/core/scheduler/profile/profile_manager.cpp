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

#include "profile_manager.h"

#include <absl/time/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>

#include "common/global_flags.h"
#include "common/rec_model_utils.h"
#include "framework/batch/batch_factory.h"
#include "framework/block/hierarchy_block_manager_pool.h"
#include "framework/request/request_state.h"

namespace xllm {

ProfileManager::ProfileManager(Engine* engine, const Options& options)
    : options_(options), engine_(engine) {
  CHECK(engine_ != nullptr);
  block_manager_pool_ = engine_->block_manager_pool();
  CHECK(block_manager_pool_ != nullptr);
  prefill_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), true /*is_prefill*/);
  decode_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), false /*is_prefill*/);
  if (options.enable_profile_step_time()) {
    LOG(INFO) << "Starting profiliing step time.";
    profile_step_time(false);
    // test accuracy
    // eval_sequence_latency_prediction();
    // eval_batch_latency_prediction("only_prefill");
    // eval_batch_latency_prediction("only_decode");
    // eval_batch_latency_prediction("mix");
  }
  if (options.enable_profile_token_budget()) {
    LOG(INFO) << "Starting profiliing token budget.";
    profile_token_budget();
  }
  if (FLAGS_enable_h2d_layerwise_copy_profile) {
    LOG(INFO) << "Starting synthetic H2D layer-wise copy profile.";
    profile_h2d_layerwise_copy();
  }
  // more profile here, such as token_budget profile and decode length
  // prediction.

#if defined(USE_NPU) || defined(USE_CUDA)
  // Warmup ACL graph executor if enabled
  if (FLAGS_enable_graph) {
    if (!is_rec_multi_round_mode()) {
      LOG(INFO) << "Starting ACL Graph/CUDA Graph warmup.";
      warmup_for_graph();
    }
  }
#endif
}

// --------------------- for test only ---------------------------
void ProfileManager::eval_sequence_latency_prediction() {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;
  int32_t token_step = 500;
  int32_t prefix_step = 500;
  int32_t upper_bound = 4000;

  LOG(INFO) << "Starting testing sequence latency prediction";
  for (int32_t token_length = token_step; token_length < upper_bound;
       token_length += token_step) {
    for (int32_t prefix_length = 0; prefix_length < token_length;
         prefix_length += prefix_step) {
      target_vec.emplace_back(run_request(token_length, prefix_length));
      pred_vec.emplace_back(predict_step_time(token_length, prefix_length));
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
void ProfileManager::eval_batch_latency_prediction(const std::string mode) {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;

  LOG(INFO) << "Starting testing batch latency prediction for " << mode;
  if (mode == "only_prefill") {
    int32_t max_batch_size = 10;
    int32_t token_step = 500;
    int32_t prefix_step = 500;
    int32_t upper_bound = 4000;
    for (int32_t token_length = token_step; token_length < upper_bound;
         token_length += token_step) {
      for (int32_t prefix_length = 0; prefix_length < token_length;
           prefix_length += prefix_step) {
        target_vec.emplace_back(
            run_request(token_length, prefix_length, max_batch_size));
        pred_vec.emplace_back(
            predict_step_time(token_length, prefix_length, max_batch_size));
      }
    }
  }
  if (mode == "only_decode") {
    int32_t max_batch_size = 200;
    int32_t token_length = 500;
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size++) {
      target_vec.emplace_back(
          run_request(token_length, token_length - 1, batch_size));
      pred_vec.emplace_back(
          predict_step_time(token_length, token_length - 1, batch_size));
    }
  }
  if (mode == "mix") {
    if (!FLAGS_enable_chunked_prefill) {
      LOG(WARNING) << "When chunked prefill is disabled, mixed prefill and "
                      "decode scenarios will not be tested.";
      return;
    }
    int32_t max_batch_size = 100;
    int32_t max_prefill_cnt = 5;
    int32_t token_length = 500;
    for (int32_t batch_size = 50; batch_size <= max_batch_size;
         batch_size += 10) {
      for (int32_t prefill_cnt = 0; prefill_cnt <= max_prefill_cnt;
           prefill_cnt++) {
        std::vector<int32_t> token_length_vec;
        std::vector<int32_t> prefix_length_vec;
        token_length_vec.insert(
            token_length_vec.end(), prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(), prefill_cnt, 0);
        // token_length_vec.insert(token_length_vec.end(), batch_size/5,
        // token_length); prefix_length_vec.insert(prefix_length_vec.end(),
        // batch_size/5, token_length-1);
        token_length_vec.insert(
            token_length_vec.end(), batch_size - prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(),
                                 batch_size - prefill_cnt,
                                 token_length - 1);
        target_vec.emplace_back(
            run_request(token_length_vec, prefix_length_vec));
        pred_vec.emplace_back(
            predict_step_time(token_length_vec, prefix_length_vec));
      }
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
// -------------------------------------------------------------

// ---------------------- dump to file-----------------------
std::string ProfileManager::generate_filename(const std::string& file_suffix) {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

  std::string filename;
  filename = ss.str() + "_" + file_suffix + ".txt";

  return filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::pair<int32_t, double>>& time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << data.first << "," << data.second << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << std::get<0>(data) << "," << std::get<1>(data) << ","
            << std::get<2>(data) << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}
// -------------------------------------------------------------

void ProfileManager::profile_step_time(bool if_dump_to_file) {
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  // TODO: support length for decode request profile
  int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  auto block_size = block_manager_pool_->options().block_size();
  bool enable_profile_kv_blocks = options_.enable_profile_kv_blocks();

  // warm up
  run_request(profile_max_prompt_length, 0);

  // prefill time profile
  if (options_.enable_profile_kv_blocks()) {
    // starting from max_context_len, dividing the token length by 2 in
    // each loop iteration
    // consider to generate kv blocks for prompt
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      // increase prefix length according to block size
      auto block_step = (profile_length_step_ + block_size - 1) / block_size;
      for (int32_t prefix_length = 0;
           prefix_length < token_length - 1 + (block_step * block_size);
           prefix_length += (block_step * block_size)) {
        if (prefix_length > token_length - 1) {
          // avoid kv_cache_token_num == token_length
          prefix_length = token_length - 1;
        }
        double latency_mean = 0;

        for (int32_t k = 0; k < profile_count_per_step_; k++) {
          latency_mean += run_request(token_length, prefix_length);
        }
        latency_mean /= profile_count_per_step_;
        // use token_length and prefix_length to predict
        time_profiling_data.emplace_back(
            token_length, prefix_length, latency_mean);
      }
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  } else {
    // not consider kv cache
    std::vector<std::pair<int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length *= 0.8) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, 0);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, latency_mean);
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  }
  if (FLAGS_enable_disagg_pd) {
    LOG(INFO) << "Disagg PD enabled, skip decode time profile.";
    return;
  }
  // decode time profile

  std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
  int32_t max_batch_size = 25;
  // for (int32_t token_length = profile_max_prompt_length; token_length >
  // 1;token_length >>= 1)
  for (int32_t token_length = 2; token_length < profile_max_prompt_length;
       token_length += profile_length_step_) {
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size += 2) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, token_length - 1, batch_size);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, batch_size, latency_mean);
    }
  }
  if (if_dump_to_file) {
    dump_step_time_profile_to_file(time_profiling_data, false /*is_prefill*/);
  }
  train_decode_time_predictor(time_profiling_data);
}

void ProfileManager::train_prefill_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_prefill_time_predictor(
    std::vector<std::pair<int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_decode_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  decode_time_predictor_->fit_for_decode(time_profiling_data);
}

// ----------------------predict step time-----------------------
std::vector<double> ProfileManager::get_coefficients(bool is_prefill) {
  if (is_prefill) {
    return prefill_time_predictor_->get_coefficients();
  } else {
    return decode_time_predictor_->get_coefficients();
  }
}

double ProfileManager::get_constant_overhead() {
  if (prefill_time_predictor_->is_trained() &&
      decode_time_predictor_->is_trained()) {
    return (prefill_time_predictor_->get_constant_overhead() +
            decode_time_predictor_->get_constant_overhead()) /
           2;
  } else if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_constant_overhead();
  } else if (decode_time_predictor_->is_trained()) {
    return decode_time_predictor_->get_constant_overhead();
  }
  return 0.0;
}

int32_t ProfileManager::get_quadratic_root(Sequence* sequence, double budget) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_state().kv_cache_tokens_num();
  if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_quadratic_root(prefix_length, budget);
  }
  LOG(ERROR) << "Prefill time predictor is not trained yet.";
  return 0;
}

// for single sequence
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  CHECK(length > prefix_length)
      << "Token length (" << length << ") must be greater than prefix length "
      << " (" << prefix_length << ").";
  double ratio = 1.0;
  if (force_use_prefill_predictor) {
    return ratio * prefill_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  }
  if (length - 1 == prefix_length) {
    return ratio * decode_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  } else {
    return ratio * prefill_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  }
}

double ProfileManager::predict_step_time(Sequence* sequence,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_cache_tokens_num();
  double latency = predict_step_time(length,
                                     prefix_length,
                                     if_need_add_constant_term,
                                     force_use_prefill_predictor);
  return latency;
}
// for single batch or sequences
double ProfileManager::predict_step_time(
    const std::vector<int32_t>& length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(length_vec.size() == prefix_length_vec.size());
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < length_vec.size(); i++) {
    // predict for each sequence
    int32_t length = length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}

// for seq in batch with the same token and prefix length
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         int32_t batch_size) {
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < batch_size; i++) {
    // predict for each sequence
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}
// ---------------------------------------------

// ----------------------for profile token budget-----------------------
void ProfileManager::profile_token_budget() {
  // use token budget means defaultly ignoring prefix cache and decode request's
  // kv cache load overhead
  // warm up
  run_request(options_.profile_max_prompt_length(), 0);
  profile_token_budget_ =
      binary_search_max_tokens(options_.max_global_tpot_ms(), 1, 4096);
  LOG(INFO) << "Profile token budget: " << profile_token_budget_
            << "for TPOT SLO: " << options_.max_global_tpot_ms();
}

bool ProfileManager::check_if_satisfy_slo(int32_t num_tokens,
                                          int32_t tpot_slo_ms) {
  // int32_t prompt_tokens_per_batch = 1024;
  // auto batch_size = num_tokens / prompt_tokens_per_batch;
  // int32_t extra_token_length = num_tokens % prompt_tokens_per_batch;
  // double batch_latency = 0;
  // for (int32_t k = 0; k < profile_count_per_step_; k++) {
  //   batch_latency +=
  //       run_request(prompt_tokens_per_batch, 0, batch_size,
  //       extra_token_length);
  // }
  double batch_latency = 0;
  for (int32_t k = 0; k < profile_count_per_step_; k++) {
    batch_latency += run_request(num_tokens, 0, 1, 0);
  }
  batch_latency /= profile_count_per_step_;
  if (batch_latency <= tpot_slo_ms) {
    return true;
  } else {
    return false;
  }
}

int32_t ProfileManager::binary_search_max_tokens(int32_t tpot_slo_ms,
                                                 int32_t lower_bound,
                                                 int32_t upper_bound) {
  int32_t left = lower_bound;
  int32_t right = upper_bound;
  // [left, right)
  while (left < right) {
    int32_t mid = left + (right - left) / 2;
    if (check_if_satisfy_slo(mid, tpot_slo_ms)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

int32_t ProfileManager::get_token_budget() { return profile_token_budget_; }

// ---------------------------------------------

const std::vector<ProfileManager::CopyBlockProfile>&
ProfileManager::get_copy_block_profile() {
  // NOTE: Add more model profiles here
  static const std::vector<CopyBlockProfile> profiles = {
      // offline copy block profile
      // {"Qwen2-7B", 128, -1, 0.48, 0.24, "Qwen2-7B, block_size=128"},
      // {"Qwen2-7B", 128, -1, 0.36, 0.74, "Qwen2-7B, block_size=128"},
      {"Qwen2-7B", 128, -1, 0.8, 0.74, "Qwen2-7B, block_size=128"},
      {"Qwen2-7B", 64, -1, 0.20, 0.25, "Qwen2-7B, block_size=64"},
      {"Qwen3-32B", 128, 2, 0.972, 0.14, "Qwen3-32B, block_size=128, tp=2"},
      {"Qwen3-32B", 128, 4, 0.588, 0.14, "Qwen3-32B, block_size=128, tp=4"},
  };

  return profiles;
}

const ProfileManager::CopyBlockProfile* ProfileManager::find_profile(
    const std::string& model_name,
    int32_t block_size,
    int32_t tp_size) const {
  auto to_lower = [](const std::string& input) {
    std::string output = input;
    std::transform(
        output.begin(), output.end(), output.begin(), [](unsigned char c) {
          return std::tolower(c);
        });
    return output;
  };
  const std::string model_name_lower = to_lower(model_name);
  const auto& profiles = get_copy_block_profile();
  for (const auto& profile : profiles) {
    const std::string profile_name_lower = to_lower(profile.model_name);
    const bool match_model =
        (profile_name_lower == model_name_lower) ||
        (model_name_lower.find(profile_name_lower) != std::string::npos);
    const bool match_tp = (profile.tp_size <= 0 || profile.tp_size == tp_size);
    if (match_model && profile.block_size == block_size && match_tp) {
      return &profile;
    }
  }
  LOG(ERROR) << "No profile found for " << model_name
             << " with block_size=" << block_size << ", tp_size=" << tp_size
             << ", using default values";
  return nullptr;
}

int32_t ProfileManager::get_max_copy_block_num(double latency_budget) {
  auto block_size = block_manager_pool_->options().block_size();
  const int32_t tp_size = std::max(options_.tp_size(), 1);
  const CopyBlockProfile* profile =
      find_profile(FLAGS_model_id, block_size, tp_size);

  double a = 1, b = 0;  // default values
  if (profile) {
    a = profile->slope;
    b = profile->intercept;
  }

  double max_blocks = std::max((latency_budget - b) / a, 0.0);
  return static_cast<int32_t>(max_blocks);
}

double ProfileManager::predict_copy_blocks_time(
    size_t num_copy_blocks,
    bool if_need_add_constant_term) {
  auto block_size = block_manager_pool_->options().block_size();
  const int32_t tp_size = std::max(options_.tp_size(), 1);
  const CopyBlockProfile* profile =
      find_profile(FLAGS_model_id, block_size, tp_size);

  double a = 1, b = 0;  // default values
  if (profile) {
    a = profile->slope;
    b = profile->intercept;
  }
  return if_need_add_constant_term ? a * num_copy_blocks + b
                                   : a * num_copy_blocks;
}

std::shared_ptr<Request> ProfileManager::generate_single_request(
    int32_t token_length,
    int32_t prefix_length,
    int32_t seq_capacity,
    bool allocate_kv_blocks) {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t eos_token_id = model_args.eos_token_id();

  std::random_device rd;
  std::mt19937_64 gen(rd());

  // If req_state does not initialize the stopchecker, default eos_token_id = 0,
  // need to skip it
  std::uniform_int_distribution<int32_t> dis(1, vocab_size - 2);

  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() {
    int32_t token = dis(gen);
    return token == eos_token_id ? token + 1 : token;  // skip eos
  });

  RequestState req_state(token_ids);
  if (seq_capacity > 0) {
    req_state.seq_capacity = seq_capacity;
  }
  req_state.enable_schedule_overlap = options_.enable_schedule_overlap();
  auto request = std::make_shared<Request>(
      /*request_id=*/"",
      /*x_request_id=*/"",
      /*x_request_time=*/"",
      req_state);

  if (!allocate_kv_blocks) {
    return request;
  }

  // TODO: better disable prefix cache
  if (prefix_length > 0) {
    if (!block_manager_pool_->BlockManagerPool::allocate(
            request->sequences()[0].get(), prefix_length)) {
      LOG(FATAL) << "Profiling time failed! Not enough blocks, prefix length : "
                 << prefix_length;
    }
    request->sequences()[0]->kv_state().incr_kv_cache_tokens_num(prefix_length);
  }

  if (!block_manager_pool_->BlockManagerPool::allocate(
          request->sequences()[0].get(), token_length)) {
    LOG(FATAL) << "Profiling time failed! Not enough blocks, token length : "
               << token_length;
  }

  return request;
}

void ProfileManager::profile_h2d_layerwise_copy() {
  const auto& block_options = block_manager_pool_->options();
  if (block_options.host_num_blocks() <= block_options.num_blocks()) {
    LOG(WARNING) << "Skip synthetic H2D layer-wise copy profile because host "
                    "KV cache is not enabled. Set host_blocks_factor > 1.";
    return;
  }
  CHECK(dynamic_cast<HierarchyBlockManagerPool*>(block_manager_pool_) !=
        nullptr)
      << "Synthetic H2D profile requires HierarchyBlockManagerPool.";

  const bool old_enable_layer_exec_profile = FLAGS_enable_layer_exec_profile;
  const int32_t old_layer_exec_profile_log_interval =
      FLAGS_layer_exec_profile_log_interval;
  const bool old_enable_h2d_overlap_profile = FLAGS_enable_h2d_overlap_profile;
  const int32_t old_h2d_overlap_profile_log_interval =
      FLAGS_h2d_overlap_profile_log_interval;
  FLAGS_enable_layer_exec_profile =
      FLAGS_h2d_layerwise_copy_profile_enable_layer_exec_profile;
  if (FLAGS_enable_layer_exec_profile) {
    FLAGS_layer_exec_profile_log_interval = 1;
  }
  FLAGS_enable_h2d_overlap_profile = true;
  FLAGS_h2d_overlap_profile_log_interval = 1;

  struct H2DLayerwiseCopyProfileCase {
    std::string mode_name;
    uint32_t layers_wise_copy_batchs;
    bool use_h2d;
    bool recompute;
    int32_t tail_recompute_blocks_per_request;
  };
  const uint32_t configured_layers_wise_copy_batchs =
      std::max<uint32_t>(1, FLAGS_layers_wise_copy_batchs);
  std::vector<H2DLayerwiseCopyProfileCase> profile_cases;
  if (configured_layers_wise_copy_batchs == 1) {
    profile_cases.push_back({"device_cached", 0, false, false, 0});
    profile_cases.push_back({"recompute", 0, false, true, 0});
    profile_cases.push_back({"full_wait", 1, true, false, 0});
    profile_cases.push_back({"full_wait_minus1", 1, true, false, 1});
  } else {
    profile_cases.push_back(
        {"layer_wise", configured_layers_wise_copy_batchs, true, false, 0});
    profile_cases.push_back({"layer_wise_minus1",
                             configured_layers_wise_copy_batchs,
                             true,
                             false,
                             1});
  }
  const int32_t warmup_steps =
      std::max(0, FLAGS_h2d_layerwise_copy_profile_warmup_steps);
  const int32_t profile_steps =
      std::max(1, FLAGS_h2d_layerwise_copy_profile_steps);

  for (const auto& profile_case : profile_cases) {
    LOG(INFO) << "[h2d_layerwise_copy_profile] mode_begin="
              << profile_case.mode_name << ", layers_wise_copy_batchs="
              << profile_case.layers_wise_copy_batchs
              << ", server_layers_wise_copy_batchs="
              << configured_layers_wise_copy_batchs
              << ", warmup_steps=" << warmup_steps
              << ", profile_steps=" << profile_steps
              << ", enable_control_h2d_block_num="
              << FLAGS_enable_control_h2d_block_num
              << ", enable_layer_exec_profile="
              << FLAGS_enable_layer_exec_profile;

    for (int32_t step = 0; step < warmup_steps; ++step) {
      run_h2d_layerwise_copy_profile_batch(
          profile_case.mode_name,
          profile_case.layers_wise_copy_batchs,
          profile_case.use_h2d,
          profile_case.recompute,
          profile_case.tail_recompute_blocks_per_request,
          step,
          true);
    }

    double total_wall_batch_latency_ms = 0.0;
    for (int32_t step = 0; step < profile_steps; ++step) {
      total_wall_batch_latency_ms += run_h2d_layerwise_copy_profile_batch(
          profile_case.mode_name,
          profile_case.layers_wise_copy_batchs,
          profile_case.use_h2d,
          profile_case.recompute,
          profile_case.tail_recompute_blocks_per_request,
          step,
          false);
    }

    LOG(INFO) << "[h2d_layerwise_copy_profile] mode_summary="
              << profile_case.mode_name << ", layers_wise_copy_batchs="
              << profile_case.layers_wise_copy_batchs
              << ", measured_batches=" << profile_steps
              << ", avg_wall_batch_latency_ms="
              << total_wall_batch_latency_ms / profile_steps;
  }

  FLAGS_enable_layer_exec_profile = old_enable_layer_exec_profile;
  FLAGS_layer_exec_profile_log_interval = old_layer_exec_profile_log_interval;
  FLAGS_enable_h2d_overlap_profile = old_enable_h2d_overlap_profile;
  FLAGS_h2d_overlap_profile_log_interval = old_h2d_overlap_profile_log_interval;
}

double ProfileManager::run_h2d_layerwise_copy_profile_batch(
    const std::string& mode_name,
    uint32_t layers_wise_copy_batchs,
    bool use_h2d,
    bool recompute,
    int32_t tail_recompute_blocks_per_request,
    int32_t step,
    bool warmup) {
  const auto& model_args = engine_->model_args();
  const auto& block_options = block_manager_pool_->options();
  auto* hierarchy_block_manager_pool =
      dynamic_cast<HierarchyBlockManagerPool*>(block_manager_pool_);
  CHECK(hierarchy_block_manager_pool != nullptr)
      << "Synthetic H2D profile requires HierarchyBlockManagerPool.";
  CHECK_GT(block_options.block_size(), 0)
      << "Synthetic H2D profile requires positive block_size.";
  const int32_t requested_batch_size =
      std::max(1, FLAGS_h2d_layerwise_copy_profile_batch_size);

  int32_t requested_context_len =
      std::max(1, FLAGS_h2d_layerwise_copy_profile_context_len);
  if (model_args.max_position_embeddings() > 1) {
    requested_context_len = std::min<int32_t>(
        requested_context_len, model_args.max_position_embeddings() - 1);
  }

  const int32_t requested_h2d_blocks =
      std::max(1, FLAGS_h2d_layerwise_copy_profile_h2d_blocks);
  const size_t requested_context_blocks =
      (static_cast<size_t>(requested_context_len) + block_options.block_size() -
       1) /
      block_options.block_size();
  const bool requested_context_fits =
      requested_context_blocks <= static_cast<size_t>(requested_h2d_blocks);
  const size_t context_blocks = requested_context_fits
                                    ? requested_context_blocks
                                    : static_cast<size_t>(requested_h2d_blocks);
  const int32_t batch_size =
      requested_context_fits
          ? std::min<int32_t>(
                requested_batch_size,
                static_cast<int32_t>(requested_h2d_blocks / context_blocks))
          : 1;
  const int32_t context_len = static_cast<int32_t>(std::min<size_t>(
      requested_context_len,
      context_blocks * static_cast<size_t>(block_options.block_size())));
  const size_t tail_recompute_blocks =
      use_h2d ? std::min<size_t>(static_cast<size_t>(std::max(
                                     0, tail_recompute_blocks_per_request)),
                                 context_blocks)
              : 0;
  const size_t h2d_blocks_per_request =
      use_h2d ? context_blocks - tail_recompute_blocks : 0;
  const int32_t h2d_tokens_per_request = static_cast<int32_t>(
      std::min<size_t>(static_cast<size_t>(context_len),
                       h2d_blocks_per_request *
                           static_cast<size_t>(block_options.block_size())));
  const int32_t tail_recompute_tokens_per_request =
      use_h2d ? context_len - h2d_tokens_per_request : 0;
  const int32_t expected_h2d_blocks =
      use_h2d ? static_cast<int32_t>(h2d_blocks_per_request * batch_size) : 0;
  const size_t tokens_budget =
      recompute ? static_cast<size_t>(context_len)
                : (use_h2d ? static_cast<size_t>(std::max(
                                 1, tail_recompute_tokens_per_request + 1))
                           : 1);
  CHECK_GT(batch_size, 0);
  CHECK_GT(context_len, 0);
  CHECK_GT(context_blocks, 0);

  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(batch_size);
  sequences_budget.reserve(batch_size);
  requests.reserve(batch_size);

  const int32_t eos_token_id = model_args.eos_token_id();
  int32_t decode_token_id = 1;
  if (decode_token_id == eos_token_id && model_args.vocab_size() > 2) {
    decode_token_id = 2;
  }

  for (int32_t i = 0; i < batch_size; ++i) {
    auto request = generate_single_request(
        context_len, 0, context_len + 8, /*allocate_kv_blocks=*/false);
    auto* sequence = request->sequences()[0].get();
    CHECK_EQ(sequence->kv_state().num_kv_blocks(), 0);
    CHECK_EQ(sequence->kv_state().kv_cache_tokens_num(), 0);
    const int32_t device_token_capacity =
        recompute ? context_len : context_len + 1;
    if (!hierarchy_block_manager_pool->allocate_device_blocks_for_profile(
            sequence, device_token_capacity)) {
      LOG(FATAL) << "Synthetic H2D profile failed to allocate device KV "
                    "slots, context_len="
                 << context_len;
    }
    CHECK_EQ(sequence->kv_state().kv_cache_tokens_num(), 0);
    if (use_h2d) {
      if (!hierarchy_block_manager_pool->allocate_host_blocks_for_profile(
              sequence, context_len)) {
        LOG(FATAL) << "Synthetic H2D profile failed to allocate host KV "
                      "blocks, context_len="
                   << context_len;
      }
      CHECK_EQ(sequence->kv_state().kv_cache_tokens_num(), 0);
      CHECK_EQ(sequence->host_kv_state().kv_cache_tokens_num(),
               static_cast<size_t>(context_len));
    } else {
      CHECK_EQ(sequence->host_kv_state().kv_cache_tokens_num(), 0);
    }

    requests.emplace_back(request);
    sequences.emplace_back(sequence);
    sequences_budget.emplace_back(tokens_budget);
  }

  std::vector<std::vector<BlockTransferInfo>> h2d_infos_by_dp(
      options_.dp_size());
  int32_t total_h2d_blocks = 0;

  if (use_h2d) {
    for (size_t block_idx = 0; block_idx < h2d_blocks_per_request;
         ++block_idx) {
      for (auto* sequence : sequences) {
        const int32_t dp_rank = sequence->dp_rank();
        CHECK_GE(dp_rank, 0);
        CHECK_LT(dp_rank, options_.dp_size());
        const auto hbm_blocks = sequence->kv_state().kv_blocks();
        const auto host_blocks = sequence->host_kv_state().kv_blocks();
        CHECK_LT(block_idx, hbm_blocks.size());
        CHECK_LT(block_idx, host_blocks.size());
        const int32_t src_block_id = host_blocks[block_idx].id();
        const int32_t dst_block_id = hbm_blocks[block_idx].id();
        h2d_infos_by_dp[dp_rank].emplace_back(
            src_block_id,
            dst_block_id,
            host_blocks[block_idx].get_immutable_hash_value(),
            TransferType::H2D);
        ++total_h2d_blocks;
      }
    }
  }
  CHECK_EQ(total_h2d_blocks, expected_h2d_blocks);

  if (!recompute) {
    for (auto* sequence : sequences) {
      sequence->kv_state().set_kv_cache_tokens_num(context_len);
      sequence->append_token(decode_token_id);
      if (use_h2d) {
        sequence->kv_state().set_kv_cache_tokens_num(h2d_tokens_per_request);
      }
    }
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  if (use_h2d) {
    for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
      if (h2d_infos_by_dp[dp_rank].empty()) {
        continue;
      }
      CHECK(!batches[dp_rank].empty())
          << "Synthetic H2D profile generated H2D infos for an empty DP batch.";
      batches[dp_rank].set_batch_id();
    }
  }
  const uint64_t profile_batch_id =
      batches.empty() ? 0 : batches.front().batch_id();

  const int32_t device_kv_tokens_before_h2d =
      use_h2d ? 0 : ((!recompute) ? context_len : 0);
  const int32_t host_kv_tokens = use_h2d ? context_len : 0;
  const int32_t device_kv_tokens_for_forward =
      recompute ? 0 : (use_h2d ? h2d_tokens_per_request : context_len);

  LOG(INFO) << "[h2d_layerwise_copy_profile] mode=" << mode_name
            << ", layers_wise_copy_batchs=" << layers_wise_copy_batchs
            << ", step=" << step << ", warmup=" << warmup
            << ", batch_id=" << profile_batch_id
            << ", requested_batch_size=" << requested_batch_size
            << ", batch_size=" << batch_size
            << ", requested_context_len=" << requested_context_len
            << ", context_len=" << context_len
            << ", requested_context_blocks=" << requested_context_blocks
            << ", context_blocks=" << context_blocks
            << ", requested_h2d_blocks=" << requested_h2d_blocks
            << ", h2d_blocks=" << total_h2d_blocks
            << ", h2d_blocks_per_request=" << h2d_blocks_per_request
            << ", tail_recompute_blocks_per_request=" << tail_recompute_blocks
            << ", h2d_tokens_per_request=" << h2d_tokens_per_request
            << ", tail_recompute_tokens_per_request="
            << tail_recompute_tokens_per_request
            << ", tokens_budget_per_request=" << tokens_budget
            << ", device_kv_tokens_before_h2d=" << device_kv_tokens_before_h2d
            << ", host_kv_tokens=" << host_kv_tokens
            << ", device_kv_tokens_for_forward="
            << device_kv_tokens_for_forward;

  if (use_h2d) {
    for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
      if (h2d_infos_by_dp[dp_rank].empty()) {
        continue;
      }
      engine_->transfer_kv_blocks(
          dp_rank, batches[dp_rank].batch_id(), h2d_infos_by_dp[dp_rank]);
    }

    // Profile-only guard: transfer_kv_blocks schedules the H2D copy
    // asynchronously, and the copy thread creates/registers the layer-wise
    // synchronizer. Give it a short head start so forward does not enter before
    // the synchronizer is visible.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  const absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  const double wall_batch_latency_ms =
      absl::ToDoubleMilliseconds(absl::Now() - start_time);

  LOG(INFO) << "[h2d_layerwise_copy_profile] mode=" << mode_name
            << ", layers_wise_copy_batchs=" << layers_wise_copy_batchs
            << ", step=" << step << ", warmup=" << warmup
            << ", batch_id=" << profile_batch_id
            << ", wall_batch_latency_ms=" << wall_batch_latency_ms
            << ", h2d_blocks=" << total_h2d_blocks;

  for (auto& request : requests) {
    hierarchy_block_manager_pool
        ->deallocate_host_and_device_without_cache_for_profile(
            request->sequences()[0].get());
  }

  return wall_batch_latency_ms;
}

// collect the latency of each step
double ProfileManager::run_request(int32_t token_length,
                                   int32_t prefix_length,
                                   int32_t batch_size,
                                   int32_t extra_token_length) {
  CHECK(token_length >= prefix_length);
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;

  // batch sequences with the same kv cahce and token length
  for (int32_t i = 0; i < batch_size; i++) {
    // generate random token ids and request
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // maybe another sequence for extra token length (< token_length) for token
  // budget profiling
  if (extra_token_length > 0) {
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(requests, sequences, sequences_budget);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

// currently for test only
double ProfileManager::run_request(
    const std::vector<int32_t>& token_length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(token_length_vec.size() == prefix_length_vec.size());
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;

  // batch sequences with the same kv cahce and token length
  for (int32_t i = 0; i < token_length_vec.size(); i++) {
    // generate random token ids and request
    int32_t token_length = token_length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];

    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

// Generate a batch of decode requests and execute it, then return the step
// latency.
double ProfileManager::profile_decode_step_time(int32_t token_length,
                                                int32_t batch_size,
                                                int32_t min_context_len,
                                                int32_t max_context_len) {
  double total_latency = 0;
  for (int32_t i = 0; i < profile_count_per_step_; ++i) {
    std::vector<int32_t> token_length_vec;
    std::vector<int32_t> prefix_length_vec;
    generate_random_decode_batch(batch_size * token_length,
                                 batch_size,
                                 min_context_len,
                                 max_context_len,
                                 token_length_vec,
                                 prefix_length_vec);
    double latency = run_request(token_length_vec, prefix_length_vec);
    total_latency += latency;
  }
  return total_latency / profile_count_per_step_;
}

// Generate a batch of random decode requests with an average length of
// token_length.
void ProfileManager::generate_random_decode_batch(
    int32_t total_length,
    int32_t batch_size,
    int32_t min_context_len,
    int32_t max_context_len,
    std::vector<int32_t>& token_length_vec,
    std::vector<int32_t>& prefix_length_vec) {
  CHECK(total_length >= batch_size * min_context_len);
  CHECK(total_length <= batch_size * max_context_len);

  token_length_vec.resize(batch_size, min_context_len);
  prefix_length_vec.resize(batch_size, min_context_len - 1);
  int remain = total_length - batch_size * min_context_len;

  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (int i = 0; i < batch_size; ++i) {
    if (remain == 0) break;

    int max = remain > (max_context_len - min_context_len)
                  ? (max_context_len - min_context_len)
                  : remain;

    std::uniform_int_distribution<int> dis(0, max);
    int add = dis(gen);
    token_length_vec[i] += add;
    prefix_length_vec[i] += add;
    remain -= add;
  }

  int idx = 0;
  while (remain > 0) {
    if (token_length_vec[idx % batch_size] < max_context_len) {
      token_length_vec[idx % batch_size] += 1;
      prefix_length_vec[idx % batch_size] += 1;
      --remain;
    }
    ++idx;
  }
}

void ProfileManager::warmup_for_graph() {
  LOG(INFO) << "Starting ACL Graph/CUDA Graph warmup with prefill and decode "
               "requests...";

  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  // Warmup parameters - align with bucket logic
  // Prefill: align max_tokens_per_batch to bucket
  int32_t prefill_tokens =
      std::min(FLAGS_max_tokens_per_batch, max_context_len);

  std::vector<int32_t> decode_seq_lens = {16};

  // Generate decode_batch_sizes aligned with bucket logic
  // For decode: n_tokens = batch_size * num_decoding_tokens (usually
  // num_decoding_tokens = 1) So batch_size directly corresponds to n_tokens
  // bucket values Bucket values: 1, 2, 4, 8, 16, then 32, 48, 64, ...
  // (multiples of 16)
  std::vector<int32_t> decode_batch_sizes = {1, 2, 4, 8, 16};
  int32_t max_seqs_per_batch = FLAGS_max_seqs_per_batch;
  // From 32 onwards, use multiples of 16 (bucket alignment)
  for (int32_t batch_size = 32; batch_size <= max_seqs_per_batch;
       batch_size += 16) {
    decode_batch_sizes.push_back(batch_size);
  }
  // Ensure max_seqs_per_batch is included if not already added
  if (decode_batch_sizes.back() != max_seqs_per_batch) {
    decode_batch_sizes.push_back(max_seqs_per_batch);
  }

  // Limit decode seq_lens to max_context_len
  for (auto& seq_len : decode_seq_lens) {
    if (seq_len > max_context_len) {
      seq_len = max_context_len;
    }
  }

  // ========== Warmup Prefill Request ==========
  LOG(INFO) << "Warming up prefill request: tokens=" << prefill_tokens;
  try {
    // Prefill: prefix_length = 0 (empty KV cache), batch_size = 10,
    // sequence_length = prefill_tokens / 10
    double latency = run_request(prefill_tokens, 0, 1);
    LOG(INFO) << "Prefill warmup completed: tokens=" << prefill_tokens
              << ", latency=" << latency << " ms";
  } catch (const std::exception& e) {
    LOG(WARNING) << "Prefill warmup failed: tokens=" << prefill_tokens
                 << ", error: " << e.what();
  }

  // ========== Warmup Decode Requests ==========
  // confict with async_schedule, so skip for now

  LOG(INFO) << "ACL Graph/CUDA Graph warmup completed";
}

}  // namespace xllm
