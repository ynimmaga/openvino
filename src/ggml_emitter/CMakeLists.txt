# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# EXPERIMENTAL, OFF by default. Executes GGUF models on a ggml backend (Vulkan / CPU / CUDA /
# Metal) using the GGUF frontend's decoder builder as the graph description.
#
# ggml is an EXTERNAL, SEPARATELY VERSIONED dependency (github.com/ggml-org/ggml, MIT). It is a
# standalone CMake project that exports ggml-config.cmake, so llama.cpp is NOT required and does
# not enter the dependency graph. Point -DGGML_DIR at an existing install, or let FetchContent
# pull the pinned tag.
#
# Only this component links ggml. openvino_gguf_frontend stays ggml-free and merely calls virtual
# methods on GraphEmitter.

set(TARGET_NAME openvino_ggml_emitter)

find_package(ggml QUIET)

if(NOT ggml_FOUND)
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        v0.23.0
        GIT_SHALLOW    TRUE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ggml)
endif()

add_library(${TARGET_NAME} SHARED src/emitter.cpp)

target_include_directories(${TARGET_NAME}
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # The emitter subclasses GraphEmitter and calls build_ggml_graph_from_gguf, both of which
    # live in the frontend's private source tree rather than its public headers.
    PRIVATE ${CMAKE_SOURCE_DIR}/src/frontends/gguf/src)

# openvino::core::dev supplies the dev-API headers (aligned_buffer, mmap_object) that the
# frontend's private quant/gguf.hpp pulls in.
target_link_libraries(${TARGET_NAME}
    PRIVATE openvino::runtime openvino::core::dev openvino::util
            openvino::frontend::gguf ggml::ggml ggml::ggml-base)

set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${OV_CPACK_COMP_CORE})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/openvino
        DESTINATION ${OV_CPACK_INCLUDEDIR} COMPONENT ${OV_CPACK_COMP_CORE_DEV})
