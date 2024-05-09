// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class PrevSequenceLengthPattern;

}  // namespace pass
}  // namespace ov

class ov::pass::PrevSequenceLengthPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PrevSequenceLengthPattern", "0");
    explicit PrevSequenceLengthPattern(const std::shared_ptr<ov::op::v1::Subtract>& prev_max_seq_len);
};