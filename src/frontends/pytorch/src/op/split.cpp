// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_chunk(const NodeContext& context) {
    // Schema: aten::chunk(Tensor input, int chunks, int dim=0) -> Tensor

    num_inputs_check(context, 2, 3);
    auto num_chunks = context.const_input<int>(1);
    auto dim = context.get_input(2);

    std::shared_ptr<ov::Node> chunk;
    chunk = context.mark_node(std::make_shared<v1::Split>(context.get_input(0), dim, num_chunks));

    return {context.mark_output(chunk)};

};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
