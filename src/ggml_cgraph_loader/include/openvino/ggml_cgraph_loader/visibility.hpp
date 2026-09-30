// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define OV_GGML_CGRAPH_LOADER_API
#else
#    ifdef openvino_ggml_cgraph_loader_EXPORTS
#        define OV_GGML_CGRAPH_LOADER_API OPENVINO_CORE_EXPORTS
#    else
#        define OV_GGML_CGRAPH_LOADER_API OPENVINO_CORE_IMPORTS
#    endif  // openvino_ggml_cgraph_loader_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
