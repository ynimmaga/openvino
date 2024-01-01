// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {
/// \brief Scaled dot product attention from PyTorch, fused with Concat
///
/// \ingroup ov_ops_cpp_api

class ScaledDotProductAttentionWithKVCache : public ov::op::Op {
public:
    OPENVINO_OP("ScaledDotProductAttentionWithKVCache", "cpu_plugin_opset");

    ScaledDotProductAttentionWithKVCache() = default;

    struct Config {
        bool output_BLHxS = false;       // true implies that output is [B,L,H*S]

        bool fuse_causal_attn = false;   // fuse causal mask and attn mask into attn_mask
        bool is_causal = false;          // apply causal mask internally
        bool fuse_concat = false;        // fuse (concat->sdp) ==> sdp
        std::vector<size_t> permute_axes; // not empty means input has transpose. output of permutation is [B,H,L,S]
                                         // e.g. [L,B,H,S] -> permute[1, 2, 0, 3] ->[B, H, L, S]
    };

    ScaledDotProductAttentionWithKVCache(const OutputVector& args, const Config& cfg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    const Config& get_config() const {
        return m_config;
    }

    Config& get_config() {
        return m_config;
    }

private:
    Config m_config;
};

}  // namespace intel_cpu
}  // namespace ov