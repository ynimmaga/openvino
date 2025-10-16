# src/frontends/onnx/provider/CMakeLists.txt

project(OpenVINOExecutionProvider)

# Enforce C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies installed via vcpkg
find_package(ONNXRuntime REQUIRED)
find_package(ONNX REQUIRED)
#find_package(GSL REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

add_definitions(-DGIT_COMMIT_ID="unknown")

# Add the OpenVINO EP library
add_library(openvino_ep SHARED
    plugin/ov_compute.cc
    plugin/ov_compute_stateful.cc
    plugin/ov_custom_allocator.cc
    plugin/ov_factory.cc
    plugin/ov_plugin_utils.cc
    plugin/ov_provider.cc
    plugin/ov_shared_context.cc
    plugin/transformations/ov_weights_as_input.cc
    common/ov_common_utils.cc
    common/ov_stateful_patch_utils.cc
    common/ov_supported_ops.cc    
)

target_include_directories(openvino_ep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(openvino_ep PUBLIC
    openvino::runtime::dev   # OpenVINO runtime internal target
    onnxruntime::onnxruntime
    Microsoft.GSL::GSL
    nlohmann_json::nlohmann_json
    ONNX::onnx
    ONNX::onnx_proto
)


target_include_directories(openvino_ep PUBLIC
    $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIR}           # <- Add this line
)

#target_include_directories(openvino_ep PRIVATE
#    ${CMAKE_SOURCE_DIR}/thirdparty/onnx
#    ${CMAKE_BINARY_DIR}/thirdparty/onnx  # where onnx_pb.h gets generated
#)
