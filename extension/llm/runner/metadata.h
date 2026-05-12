/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace metadata {

inline constexpr const char* kPrefix = "metadata.";

inline constexpr const char* kBosId = "metadata.tokenizer.bos_id";
inline constexpr const char* kEosIds = "metadata.tokenizer.eos_ids";
inline constexpr const char* kMaxSeqLen = "metadata.context.max_seq_len";
inline constexpr const char* kMaxContextLen =
    "metadata.context.max_context_len";
inline constexpr const char* kVocabSize = "metadata.model.vocab_size";
inline constexpr const char* kUseKVCache = "metadata.model.use_kv_cache";
inline constexpr const char* kNLayers = "metadata.model.n_layers";
inline constexpr const char* kChatTemplate = "metadata.tokenizer.chat_template";

inline runtime::Result<int64_t> get_int(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() != sizeof(int64_t)) {
    return runtime::Error::InvalidArgument;
  }
  int64_t value;
  std::memcpy(&value, buffer.data(), sizeof(int64_t));
  return value;
}

inline runtime::Result<double> get_float(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() != sizeof(double)) {
    return runtime::Error::InvalidArgument;
  }
  double value;
  std::memcpy(&value, buffer.data(), sizeof(double));
  return value;
}

inline runtime::Result<std::string> get_string(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  std::string value(static_cast<const char*>(buffer.data()), buffer.size());
  return value;
}

inline runtime::Result<std::vector<int64_t>> get_int_list(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() < sizeof(uint32_t)) {
    return runtime::Error::InvalidArgument;
  }
  uint32_t count;
  std::memcpy(&count, buffer.data(), sizeof(uint32_t));
  size_t payload = buffer.size() - sizeof(uint32_t);
  if (payload / sizeof(int64_t) != count || payload % sizeof(int64_t) != 0) {
    return runtime::Error::InvalidArgument;
  }
  std::vector<int64_t> values(count);
  if (count > 0) {
    std::memcpy(
        values.data(),
        static_cast<const char*>(buffer.data()) + sizeof(uint32_t),
        count * sizeof(int64_t));
  }
  return values;
}

} // namespace metadata
} // namespace llm
} // namespace extension
} // namespace executorch
