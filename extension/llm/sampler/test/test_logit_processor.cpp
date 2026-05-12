/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/sampler/logit_processor.h>

#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

using ::executorch::extension::llm::LogitProcessor;

namespace {

// Adds a fixed bias to every logit slot. Records how many times it was
// invoked so tests can verify chain ordering.
class AddBiasProcessor : public LogitProcessor {
 public:
  explicit AddBiasProcessor(float bias) : bias_(bias) {}

  void process(float* logits, int32_t vocab_size) override {
    ++call_count_;
    for (int32_t i = 0; i < vocab_size; ++i) {
      logits[i] += bias_;
    }
  }

  int call_count() const {
    return call_count_;
  }

 private:
  float bias_;
  int call_count_ = 0;
};

class MultiplyProcessor : public LogitProcessor {
 public:
  explicit MultiplyProcessor(float factor) : factor_(factor) {}

  void process(float* logits, int32_t vocab_size) override {
    for (int32_t i = 0; i < vocab_size; ++i) {
      logits[i] *= factor_;
    }
  }

 private:
  float factor_;
};

class MaskTokenProcessor : public LogitProcessor {
 public:
  explicit MaskTokenProcessor(int32_t banned_token)
      : banned_token_(banned_token) {}

  void process(float* logits, int32_t vocab_size) override {
    if (banned_token_ >= 0 && banned_token_ < vocab_size) {
      logits[banned_token_] = -std::numeric_limits<float>::infinity();
    }
  }

 private:
  int32_t banned_token_;
};

} // namespace

// A single processor sees the buffer and may mutate it in place.
TEST(LogitProcessorTest, SingleProcessorMutatesLogits) {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
  AddBiasProcessor bias{10.0f};

  bias.process(logits.data(), static_cast<int32_t>(logits.size()));

  const std::vector<float> expected = {11.0f, 12.0f, 13.0f, 14.0f};
  EXPECT_EQ(logits, expected);
  EXPECT_EQ(bias.call_count(), 1);
}

// Multiply(×2) then Add(+1) gives (x*2)+1, which differs from
// Add(+1) then Multiply(×2) = (x+1)*2. Non-commutative operations
// verify that processors run in registration order.
TEST(LogitProcessorTest, ProcessorChainAppliesInOrder) {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};

  std::vector<std::shared_ptr<LogitProcessor>> chain;
  chain.push_back(std::make_shared<MultiplyProcessor>(2.0f));
  chain.push_back(std::make_shared<AddBiasProcessor>(1.0f));

  for (auto& p : chain) {
    // NOLINTNEXTLINE(facebook-hte-Deprecated)
    p->process(logits.data(), static_cast<int32_t>(logits.size()));
  }

  // (x*2)+1, NOT (x+1)*2
  const std::vector<float> expected = {3.0f, 5.0f, 7.0f, 9.0f};
  EXPECT_EQ(logits, expected);
}

// A masking processor zeroes (well, -inf's) a specific token slot. This is
// the pattern grammar processors will follow.
TEST(LogitProcessorTest, MaskTokenDrivesArgmaxAway) {
  std::vector<float> logits = {0.1f, 0.2f, 0.99f, 0.4f}; // argmax = 2

  MaskTokenProcessor mask{/*banned_token=*/2};
  mask.process(logits.data(), static_cast<int32_t>(logits.size()));

  const std::vector<float> expected = {
      0.1f, 0.2f, -std::numeric_limits<float>::infinity(), 0.4f};
  EXPECT_EQ(logits, expected);
}

TEST(LogitProcessorTest, MaskTokenOutOfRangeIsNoOp) {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  const std::vector<float> snapshot = logits;

  MaskTokenProcessor mask_over{/*banned_token=*/99};
  mask_over.process(logits.data(), static_cast<int32_t>(logits.size()));
  EXPECT_EQ(logits, snapshot);

  MaskTokenProcessor mask_neg{/*banned_token=*/-1};
  mask_neg.process(logits.data(), static_cast<int32_t>(logits.size()));
  EXPECT_EQ(logits, snapshot);
}
