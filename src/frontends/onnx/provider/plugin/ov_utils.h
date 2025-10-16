// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <stdexcept>
#include <optional>

#include "onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define OVEP_ERROR_STR(...) onnxruntime::openvino_ep_plugin::BuildErrorMessage(__FILE__, __LINE__, __VA_ARGS__)

#define OVEP_RETURN_IF_ERROR(fn) \
  do {                           \
    OrtStatus* _status = (fn);   \
    if (_status != nullptr) {    \
      return _status;            \
    }                            \
  } while (0)

#define OVEP_RETURN_IF(cond, ort_api, msg)                                     \
  do {                                                                         \
    if ((cond)) {                                                              \
      return (ort_api).CreateStatus(ORT_EP_FAIL, OVEP_ERROR_STR(msg).c_str()); \
    }                                                                          \
  } while (0)

#define OVEP_RETURN_STATUS_IF(cond, status, msg)                              \
  do {                                                                        \
    if ((cond)) {                                                             \
      return Ort::GetApi().CreateStatus(status, OVEP_ERROR_STR(msg).c_str()); \
    }                                                                         \
  } while (0)

#define OVEP_DISABLE_MOVE(class_name) \
  class_name(class_name&&) = delete;  \
  class_name& operator=(class_name&&) = delete;

#define OVEP_DISABLE_COPY(class_name)     \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete;

#define OVEP_DISABLE_COPY_AND_MOVE(class_name) \
  OVEP_DISABLE_COPY(class_name)                \
  OVEP_DISABLE_MOVE(class_name)

#define OVEP_THROW(...)                                    \
  do {                                                     \
    throw std::runtime_error(OVEP_ERROR_STR(__VA_ARGS__)); \
  } while (0)

#define OVEP_ENFORCE(condition, ...) \
  do {                               \
    if (!(condition)) {              \
      OVEP_THROW(__VA_ARGS__);       \
    }                                \
  } while (0)

namespace onnxruntime {
namespace openvino_ep_plugin {

constexpr const char* kOvepLogTag = "[OpenVINO-EP] ";

template <typename... Args>
static std::string BuildErrorMessage(const char* file, int line, Args&&... args) {
  std::ostringstream oss;
  oss << kOvepLogTag;
  oss << file << ":" << line << " - ";
  ((oss << args), ...);
  return std::move(oss).str();
}

template <typename Func>
static auto ApiEntry(Func&& func, std::optional<std::reference_wrapper<Ort::Logger>> logger = std::nullopt) {
  try {
    return func();
  } catch (const Ort::Exception& ex) {
    if (logger) {
      ORT_CXX_LOG_NOEXCEPT(logger->get(), ORT_LOGGING_LEVEL_ERROR, ex.what());
    }
    if constexpr (std::is_same_v<decltype(func()), OrtStatus*>) {
      return Ort::Status(ex.what(), ex.GetOrtErrorCode()).release();
    }
  } catch (const std::exception& ex) {
    if (logger) {
      ORT_CXX_LOG_NOEXCEPT(logger->get(), ORT_LOGGING_LEVEL_ERROR, ex.what());
    }
    if constexpr (std::is_same_v<decltype(func()), OrtStatus*>) {
      return Ort::Status(ex.what(), ORT_RUNTIME_EXCEPTION).release();
    }
  } catch (...) {
    if (logger) {
      ORT_CXX_LOG_NOEXCEPT(logger->get(), ORT_LOGGING_LEVEL_ERROR, "Unknown exception occurred.");
    }
    if constexpr (std::is_same_v<decltype(func()), OrtStatus*>) {
      return Ort::Status("Unknown exception occurred.", ORT_RUNTIME_EXCEPTION).release();
    }
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
