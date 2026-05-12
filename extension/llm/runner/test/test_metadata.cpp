/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/metadata.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace executorch::extension::llm::metadata;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;

class MockNamedDataMap : public NamedDataMap {
 public:
  void add(const std::string& key, std::vector<uint8_t> data) {
    data_[key] = std::move(data);
  }

  Result<FreeableBuffer> get_data(
      executorch::aten::string_view key) const override {
    std::string k(key.data(), key.size());
    auto it = data_.find(k);
    if (it == data_.end()) {
      return Error::NotFound;
    }
    return FreeableBuffer(it->second.data(), it->second.size(), nullptr);
  }

  Result<const TensorLayout> get_tensor_layout(
      executorch::aten::string_view /*key*/) const override {
    return Error::NotFound;
  }

  Error load_data_into(
      executorch::aten::string_view /*key*/,
      void* /*buffer*/,
      size_t /*size*/) const override {
    return Error::NotFound;
  }

  Result<uint32_t> get_num_keys() const override {
    return static_cast<uint32_t>(data_.size());
  }

  Result<const char*> get_key(uint32_t /*index*/) const override {
    return Error::NotFound;
  }

 private:
  std::unordered_map<std::string, std::vector<uint8_t>> data_;
};

// Encode an int64_t in little-endian (matches Python struct.pack("<q", v)).
std::vector<uint8_t> encode_int(int64_t v) {
  std::vector<uint8_t> buf(sizeof(int64_t));
  std::memcpy(buf.data(), &v, sizeof(int64_t));
  return buf;
}

// Encode a double in little-endian (matches Python struct.pack("<d", v)).
std::vector<uint8_t> encode_float(double v) {
  std::vector<uint8_t> buf(sizeof(double));
  std::memcpy(buf.data(), &v, sizeof(double));
  return buf;
}

std::vector<uint8_t> encode_string(const std::string& s) {
  return {s.begin(), s.end()};
}

// Encode int list: uint32_t count + count * int64_t (matches Python format).
std::vector<uint8_t> encode_int_list(const std::vector<int64_t>& vals) {
  uint32_t count = static_cast<uint32_t>(vals.size());
  std::vector<uint8_t> buf(sizeof(uint32_t) + count * sizeof(int64_t));
  std::memcpy(buf.data(), &count, sizeof(uint32_t));
  if (count > 0) {
    std::memcpy(
        buf.data() + sizeof(uint32_t), vals.data(), count * sizeof(int64_t));
  }
  return buf;
}

class MetadataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(MetadataTest, GetInt) {
  MockNamedDataMap map;
  map.add("metadata.tokenizer.bos_id", encode_int(128000));

  auto result = get_int(map, "metadata.tokenizer.bos_id");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), 128000);
}

TEST_F(MetadataTest, GetIntNegative) {
  MockNamedDataMap map;
  map.add("metadata.val", encode_int(-42));

  auto result = get_int(map, "metadata.val");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), -42);
}

TEST_F(MetadataTest, GetIntWrongSize) {
  MockNamedDataMap map;
  map.add("metadata.bad", {0x01, 0x02, 0x03});

  auto result = get_int(map, "metadata.bad");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

TEST_F(MetadataTest, GetIntMissing) {
  MockNamedDataMap map;
  auto result = get_int(map, "metadata.missing");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}

TEST_F(MetadataTest, GetFloat) {
  MockNamedDataMap map;
  map.add("metadata.temp", encode_float(0.7));

  auto result = get_float(map, "metadata.temp");
  ASSERT_TRUE(result.ok());
  EXPECT_DOUBLE_EQ(result.get(), 0.7);
}

TEST_F(MetadataTest, GetString) {
  MockNamedDataMap map;
  map.add("metadata.model.arch", encode_string("llama"));

  auto result = get_string(map, "metadata.model.arch");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), "llama");
}

TEST_F(MetadataTest, GetStringEmpty) {
  MockNamedDataMap map;
  map.add("metadata.empty", encode_string(""));

  auto result = get_string(map, "metadata.empty");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), "");
}

TEST_F(MetadataTest, GetIntList) {
  MockNamedDataMap map;
  map.add("metadata.tokenizer.eos_ids", encode_int_list({128009, 128001}));

  auto result = get_int_list(map, "metadata.tokenizer.eos_ids");
  ASSERT_TRUE(result.ok());
  const std::vector<int64_t> expected{128009, 128001};
  EXPECT_EQ(result.get(), expected);
}

TEST_F(MetadataTest, GetIntListEmpty) {
  MockNamedDataMap map;
  map.add("metadata.empty_list", encode_int_list({}));

  auto result = get_int_list(map, "metadata.empty_list");
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result.get().empty());
}

TEST_F(MetadataTest, GetIntListTruncatedHeader) {
  MockNamedDataMap map;
  map.add("metadata.bad", {0x01, 0x02});

  auto result = get_int_list(map, "metadata.bad");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

TEST_F(MetadataTest, GetIntListCountMismatch) {
  MockNamedDataMap map;
  // Header says 2 elements but only 1 element of payload.
  auto buf = encode_int_list({42});
  uint32_t fake_count = 2;
  std::memcpy(buf.data(), &fake_count, sizeof(uint32_t));
  map.add("metadata.bad", std::move(buf));

  auto result = get_int_list(map, "metadata.bad");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

} // namespace
