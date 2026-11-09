// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Rewrites vLLM's neox-style RoPE lowering into the form RoPEFusionGPTNEOX matches.
 *
 * vLLM rotates the two halves of x separately, with half-width cos/sin tables:
 *
 *     x1, x2 = Split(x, axis=-1, 2)
 *     Concat(x1*cos + (x2*sin)*-1, x2*cos + x1*sin, axis=-1)
 *
 * The pass replaces it with the full-width rotate-half form:
 *
 *     x1, x2 = VariadicSplit(x, axis=-1, [D/2, D/2])
 *     x*Concat(cos, cos) + Concat(x2*-1, x1)*Concat(sin, sin)
 *
 * It applies only when:
 *  - one 2-way Split/VariadicSplit cuts the static last dimension D into equal halves, and its outputs feed the
 *    four Multiplies directly (no view in between);
 *  - the Concat joins the two branches along the last axis;
 *  - cos and sin have the same shape, with a static last dimension of D/2.
 *
 * The subtraction must already be in the Multiply(-1) + Add form, so ov::pass::ConvertSubtract has to run first.
 * CommonOptimizations registers the pass only for models with the "vllm_model" rt_info flag.
 */
class TRANSFORMATIONS_API NormalizeVLLMRoPE : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeVLLMRoPE");
    NormalizeVLLMRoPE();
};

}  // namespace ov::pass
