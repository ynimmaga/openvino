// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <shared_mutex>

#include "openvino/runtime/core.hpp"
#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class SharedContext {
 public:
  SharedContext() {}
  struct SharedWeights {
    struct Metadata {
      struct Key {
        std::string name;
        bool operator==(const Key&) const = default;
      };
      struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
          return std::hash<std::string>()(key.name);
        }
      };
      struct Value {
        struct {
          std::filesystem::path location{"not_set"};
          size_t data_offset{0};
          size_t size{0};
        } serialized;

        std::shared_ptr<ov::Tensor> tensor;
      };
      using Map = std::unordered_map<Key, Value, Hash>;
      friend std::ostream& operator<<(std::ostream& right, const Metadata::Map& metadata);
      friend std::istream& operator>>(std::istream& right, Metadata::Map& metadata);
    };

    struct WeightsFile {
      OVEP_DISABLE_COPY_AND_MOVE(WeightsFile);
      WeightsFile() = delete;
      virtual ~WeightsFile() = default;
      explicit WeightsFile(std::filesystem::path filename);
      void LoadWeights(size_t file_offset, void* data, size_t size);

     private:
      std::ifstream file_;
      size_t weights_size_;
    };

    mutable std::shared_mutex mutex;
    std::unordered_map<std::filesystem::path, std::unique_ptr<WeightsFile>> weight_files;
    Metadata::Map metadata;

    OrtStatus* LoadMetaData(std::filesystem::path model_dir, const std::string& model_name);
    OrtStatus* SaveMetaData(std::filesystem::path model_dir, const std::string& model_name);
    static std::filesystem::path GetMetaDataFilePath(const std::filesystem::path& model_dir, const std::string& model_name) {
      std::filesystem::path model_name_path(model_name);
      return model_dir / (model_name_path.stem().string() + "_metadata.bin");
    }
    bool IsSharedWeight(const std::string& name) const {
      std::shared_lock lock(mutex);
      return metadata.contains(Metadata::Key{.name = name});
    }
    void AddExternalWeight(const std::string& name, size_t offset, size_t size, std::filesystem::path location) {
      Metadata::Value value;
      value.serialized.data_offset = offset;
      value.serialized.size = size;
      value.serialized.location = std::move(location);
      std::unique_lock lock(mutex);
      metadata[Metadata::Key{.name = name}] = std::move(value);
    }
    void SetSharedWeightsOnInferRequest(ov::InferRequest& ir, std::filesystem::path model_dir);
  } shared_weights;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
