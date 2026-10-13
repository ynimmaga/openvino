# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# EXPERIMENTAL. "GGML" device: runs cgraph artifacts dumped from llama.cpp / whisper.cpp on a ggml
# backend (Vulkan / CPU / ...) through openvino_ggml_cgraph_loader. Built with the loader.

if(NOT ENABLE_GGML_CGRAPH_LOADER)
    return()
endif()

set(TARGET_NAME openvino_ggml_plugin)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GGML"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR src/plugin.cpp
              ADD_CLANG_FORMAT)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_ggml_cgraph_loader)
