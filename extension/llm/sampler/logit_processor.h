/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * Interface for in-place logit transformations applied between the model's
 * forward pass and the sampler. Examples include:
 *   - Grammar / constrained-decoding masks (set disallowed tokens to -inf)
 *   - Logit bias (additive per-token bias)
 *   - Custom debug instrumentation
 *
 * A `TextTokenGenerator` may be configured with a chain of processors. They
 * are invoked in order on every decoding step, before the sampler sees the
 * logits. Each processor mutates the buffer in place; later processors
 * observe earlier processors' modifications.
 *
 * Implementations must be cheap to call repeatedly — `process()` runs on the
 * critical path of every generated token.
 */
class ET_EXPERIMENTAL LogitProcessor {
 public:
  virtual ~LogitProcessor() = default;

  /**
   * Modify logits in place for the current decoding step.
   *
   * @param logits      Mutable pointer to the logits buffer for the current
   *                    step. Must contain at least `vocab_size` elements.
   * @param vocab_size  Number of logits in the buffer (size of the model's
   *                    output vocabulary for the current step).
   */
  virtual void process(float* logits, int32_t vocab_size) = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
