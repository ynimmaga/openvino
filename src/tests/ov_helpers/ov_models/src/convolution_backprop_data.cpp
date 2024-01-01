// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeConvolutionBackpropData(const ov::Output<Node>& in,
                                                  const element::Type& type,
                                                  const std::vector<size_t>& filterSize,
                                                  const std::vector<size_t>& strides,
                                                  const std::vector<ptrdiff_t>& padsBegin,
                                                  const std::vector<ptrdiff_t>& padsEnd,
                                                  const std::vector<size_t>& dilations,
                                                  const op::PadType& autoPad,
                                                  size_t numOutChannels,
                                                  bool addBiases,
                                                  const std::vector<ptrdiff_t>& outputPadding,
                                                  const std::vector<float>& filterWeights,
                                                  const std::vector<float>& biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {static_cast<size_t>(shape[1].get_length()), numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode =
        ov::test::utils::deprecated::make_constant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    return makeConvolutionBackpropData(in,
                                       filterWeightsNode,
                                       type,
                                       strides,
                                       padsBegin,
                                       padsEnd,
                                       dilations,
                                       autoPad,
                                       addBiases,
                                       outputPadding,
                                       biasesWeights);
}

std::shared_ptr<Node> makeConvolutionBackpropData(const ov::Output<Node>& in,
                                                  const ov::Output<Node>& weights,
                                                  const element::Type& type,
                                                  const std::vector<size_t>& strides,
                                                  const std::vector<ptrdiff_t>& padsBegin,
                                                  const std::vector<ptrdiff_t>& padsEnd,
                                                  const std::vector<size_t>& dilations,
                                                  const op::PadType& autoPad,
                                                  bool addBiases,
                                                  const std::vector<ptrdiff_t>& outputPadding,
                                                  const std::vector<float>& biasesWeights) {
    auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                        weights,
                                                                        strides,
                                                                        padsBegin,
                                                                        padsEnd,
                                                                        dilations,
                                                                        autoPad);

    if (!outputPadding.empty()) {
        deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                       weights,
                                                                       strides,
                                                                       padsBegin,
                                                                       padsEnd,
                                                                       dilations,
                                                                       autoPad,
                                                                       outputPadding);
    }

    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = ov::test::utils::deprecated::make_constant(type, {}, biasesWeights, randomBiases);
        auto add = std::make_shared<ov::op::v1::Add>(deconv, biasesWeightsNode);
        return add;
    } else {
        return deconv;
    }
}

std::shared_ptr<Node> makeConvolutionBackpropData(const ov::Output<Node>& in,
                                                  const ov::Output<Node>& outputShape,
                                                  const element::Type& type,
                                                  const std::vector<size_t>& filterSize,
                                                  const std::vector<size_t>& strides,
                                                  const std::vector<ptrdiff_t>& padsBegin,
                                                  const std::vector<ptrdiff_t>& padsEnd,
                                                  const std::vector<size_t>& dilations,
                                                  const op::PadType& autoPad,
                                                  size_t numOutChannels,
                                                  bool addBiases,
                                                  const std::vector<ptrdiff_t>& outputPadding,
                                                  const std::vector<float>& filterWeights,
                                                  const std::vector<float>& biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {static_cast<size_t>(shape[1].get_length()), numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode =
        ov::test::utils::deprecated::make_constant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                        filterWeightsNode,
                                                                        outputShape,
                                                                        strides,
                                                                        padsBegin,
                                                                        padsEnd,
                                                                        dilations,
                                                                        autoPad);

    if (!outputPadding.empty()) {
        deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(in,
                                                                       filterWeightsNode,
                                                                       outputShape,
                                                                       strides,
                                                                       padsBegin,
                                                                       padsEnd,
                                                                       dilations,
                                                                       autoPad,
                                                                       outputPadding);
    }

    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = ov::test::utils::deprecated::make_constant(type, {}, biasesWeights, randomBiases);
        auto add = std::make_shared<ov::op::v1::Add>(deconv, biasesWeightsNode);
        return add;
    } else {
        return deconv;
    }
}

}  // namespace builder
}  // namespace ngraph
