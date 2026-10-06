// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The public GgmlModel surface shared by every graph source: backend/buffer plumbing, input and
// output access, weight/metadata read-back. The graph itself is produced elsewhere -- here by the
// cgraph loader (cgraph_loader.cpp), which replays a topology llama.cpp described offline.
//
// This file links ggml; consumers of the public header do not.

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"

#include "model_impl.hpp"

namespace ov {
namespace ggml_cgraph_loader {

GgmlModel::GgmlModel() : m_impl(new Impl()) {}
GgmlModel::~GgmlModel() = default;

bool GgmlModel::compute() {
    return ggml_backend_graph_compute(m_impl->backend, m_impl->gf) == GGML_STATUS_SUCCESS;
}

bool GgmlModel::write_input(const std::string& name, const void* data, size_t bytes) {
    auto it = m_impl->out.externals.find(name);
    if (it == m_impl->out.externals.end()) {
        return false;
    }
    OPENVINO_ASSERT(bytes <= ggml_nbytes(it->second),
                    "[GGML] write_input('", name, "'): ", bytes, " bytes exceeds tensor size ",
                    ggml_nbytes(it->second));
    ggml_backend_tensor_set(it->second, data, 0, bytes);
    return true;
}

size_t GgmlModel::input_size(const std::string& name) const {
    auto it = m_impl->out.externals.find(name);
    return it == m_impl->out.externals.end() ? 0 : static_cast<size_t>(ggml_nelements(it->second));
}

std::vector<std::string> GgmlModel::input_names() const {
    std::vector<std::string> names;
    names.reserve(m_impl->out.externals.size());
    for (const auto& kv : m_impl->out.externals) {
        names.push_back(kv.first);
    }
    return names;
}

size_t GgmlModel::logits_size() const {
    return static_cast<size_t>(m_impl->out.last->ne[0]);
}

void GgmlModel::read_logits(float* out) const {
    ggml_backend_tensor_get(m_impl->out.last, out, 0, logits_size() * sizeof(float));
}

size_t GgmlModel::output_size() const {
    const ggml_tensor* t = m_impl->out.last;
    return static_cast<size_t>(t->ne[0]) * t->ne[1] * t->ne[2] * t->ne[3];
}

void GgmlModel::read_output(float* out) const {
    ggml_backend_tensor_get(m_impl->out.last, out, 0, output_size() * sizeof(float));
}

const std::map<std::string, ggml_tensor*>& GgmlModel::externals() const {
    return m_impl->out.externals;
}

size_t GgmlModel::input_nbytes(const std::string& name) const {
    auto it = m_impl->out.externals.find(name);
    return it == m_impl->out.externals.end() ? 0 : ggml_nbytes(it->second);
}

bool GgmlModel::read_input(const std::string& name, void* dst, size_t bytes) const {
    auto it = m_impl->out.externals.find(name);
    if (it == m_impl->out.externals.end() || bytes > ggml_nbytes(it->second)) {
        return false;
    }
    ggml_backend_tensor_get(it->second, dst, 0, bytes);
    return true;
}

size_t GgmlModel::weight_nbytes(const std::string& gguf_name) const {
    auto it = m_impl->wmap.find(gguf_name);
    return it == m_impl->wmap.end() ? 0 : ggml_nbytes(it->second);
}

bool GgmlModel::read_weight(const std::string& gguf_name, size_t offset_bytes, void* dst,
                            size_t nbytes) const {
    auto it = m_impl->wmap.find(gguf_name);
    if (it == m_impl->wmap.end() || offset_bytes + nbytes > ggml_nbytes(it->second)) {
        return false;
    }
    ggml_backend_tensor_get(it->second, dst, offset_bytes, nbytes);
    return true;
}

bool GgmlModel::gguf_meta_i32(const std::string& key, int32_t& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0) {
        return false;
    }
    // GGUF writers vary on the signedness of a plain integer field (e.g. image_size is UINT32
    // here); gguf_get_val_i32/u32 assert on an exact type match, so try both.
    switch (gguf_get_kv_type(m_impl->gg, id)) {
    case GGUF_TYPE_INT32:
        out = gguf_get_val_i32(m_impl->gg, id);
        return true;
    case GGUF_TYPE_UINT32:
        out = static_cast<int32_t>(gguf_get_val_u32(m_impl->gg, id));
        return true;
    default:
        return false;
    }
}

bool GgmlModel::gguf_meta_f32_array(const std::string& key, std::vector<float>& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0 || gguf_get_arr_type(m_impl->gg, id) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    const size_t n = gguf_get_arr_n(m_impl->gg, id);
    const float* data = static_cast<const float*>(gguf_get_arr_data(m_impl->gg, id));
    out.assign(data, data + n);
    return true;
}

namespace {

// Raw element data of a GGUF array of exactly `type`, or nullptr.
const void* gguf_array(gguf_context* gg, const std::string& key, gguf_type type, size_t& n) {
    if (!gg) {
        return nullptr;
    }
    const int64_t id = gguf_find_key(gg, key.c_str());
    if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gg, id) != type) {
        return nullptr;
    }
    n = gguf_get_arr_n(gg, id);
    return gguf_get_arr_data(gg, id);
}

}  // namespace

bool GgmlModel::gguf_meta_i32_array(const std::string& key, std::vector<int32_t>& out) const {
    size_t n = 0;
    const auto* data = static_cast<const int32_t*>(gguf_array(m_impl->gg, key, GGUF_TYPE_INT32, n));
    if (!data) {
        return false;
    }
    out.assign(data, data + n);
    return true;
}

bool GgmlModel::gguf_meta_u8_array(const std::string& key, std::vector<uint8_t>& out) const {
    size_t n = 0;
    const auto* data = static_cast<const uint8_t*>(gguf_array(m_impl->gg, key, GGUF_TYPE_UINT8, n));
    if (!data) {
        return false;
    }
    out.assign(data, data + n);
    return true;
}

bool GgmlModel::gguf_meta_str_array(const std::string& key, std::vector<std::string>& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0 || gguf_get_kv_type(m_impl->gg, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(m_impl->gg, id) != GGUF_TYPE_STRING) {
        return false;
    }
    const size_t n = gguf_get_arr_n(m_impl->gg, id);
    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(m_impl->gg, id, i));
    }
    return true;
}

ggml_tensor* GgmlModel::logits() const {
    return m_impl->out.last;
}

const std::set<std::string>& GgmlModel::unsupported_ops() const {
    return m_impl->out.unsupported;
}

size_t GgmlModel::node_count() const {
    return static_cast<size_t>(ggml_graph_n_nodes(m_impl->gf));
}

std::string GgmlModel::backend_name() const {
    return ggml_backend_name(m_impl->backend);
}

}  // namespace ggml_cgraph_loader
}  // namespace ov
