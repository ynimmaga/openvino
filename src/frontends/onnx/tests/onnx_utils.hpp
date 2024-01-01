// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>

#include <openvino/core/model.hpp>
#include <openvino/frontend/extension.hpp>
#include <string>

#include "common_test_utils/test_constants.hpp"

// Resolves different backend names to an internal device enumeration
inline std::string backend_name_to_device(const std::string& backend_name) {
    if (backend_name == "INTERPRETER")
        return ov::test::utils::DEVICE_TEMPLATE;
    if (backend_name == "IE_CPU")
        return ov::test::utils::DEVICE_CPU;
    if (backend_name == "IE_GPU")
        return ov::test::utils::DEVICE_GPU;
    throw "Unsupported backend name";
}

namespace ov {
namespace frontend {
namespace onnx {
namespace tests {

extern const std::string ONNX_FE;

// A wrapper to create ONNX Frontend and configure the conversion pipeline
std::shared_ptr<ov::Model> convert_model(const std::string& model_path,
                                         const ov::frontend::ConversionExtensionBase::Ptr& conv_ext = nullptr);
// A wrapper to create ONNX Frontend and configure the conversion pipeline
std::shared_ptr<ov::Model> convert_model(std::ifstream& model_stream);
// A wrapper to create ONNX Frontend and configure the conversion pipeline to get
// a model with possible Framework Nodes
std::shared_ptr<ov::Model> convert_partially(const std::string& model_path);

// Returns path to a manifest file
std::string onnx_backend_manifest(const std::string& manifest);
}  // namespace tests
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

// For compatibility purposes, need to remove when will be unused
extern const std::string ONNX_FE;
