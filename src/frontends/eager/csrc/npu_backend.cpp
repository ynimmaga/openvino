// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// npu_backend.cpp — PrivateUse1 "npu" device backend for PyTorch.
// CPU-backed allocator + device guard + copy_ and empty kernels.
// The actual op execution is done in Python via the OV frontend.
// This C++ layer just provides the device infrastructure.

#include <torch/extension.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Storage.h>

// ═══════════════════════════════════════════════════════════════════════════
// 1. CPU-backed allocator for PrivateUse1 "npu" device
// ═══════════════════════════════════════════════════════════════════════════

struct NpuAllocator final : at::Allocator {
    at::DataPtr allocate(size_t n) override {
        void* data = nullptr;
        if (n > 0) {
            data = c10::alloc_cpu(n);
        }
        return {data, data, &NpuAllocator::deleter,
                at::Device(at::DeviceType::PrivateUse1, 0)};
    }

    static void deleter(void* ptr) {
        if (ptr) {
            c10::free_cpu(ptr);
        }
    }

    at::DeleterFnPtr raw_deleter() const override {
        return &NpuAllocator::deleter;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const final {
        default_copy_data(dest, src, count);
    }
};

static NpuAllocator g_npu_allocator;

// ═══════════════════════════════════════════════════════════════════════════
// 2. Device guard (single simulated device)
// ═══════════════════════════════════════════════════════════════════════════

struct NpuGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    at::DeviceType type() const override {
        return at::DeviceType::PrivateUse1;
    }
    c10::Device exchangeDevice(c10::Device) const override {
        return c10::Device(at::DeviceType::PrivateUse1, 0);
    }
    c10::Device getDevice() const override {
        return c10::Device(at::DeviceType::PrivateUse1, 0);
    }
    void setDevice(c10::Device) const override {}
    void uncheckedSetDevice(c10::Device) const noexcept override {}
    c10::Stream getStream(c10::Device) const noexcept override {
        return c10::Stream(c10::Stream::DEFAULT,
                           c10::Device(at::DeviceType::PrivateUse1, 0));
    }
    c10::Stream getNewStream(c10::Device, int priority = 0) const override {
        (void)priority;
        return c10::Stream(c10::Stream::DEFAULT,
                           c10::Device(at::DeviceType::PrivateUse1, 0));
    }
    c10::Stream exchangeStream(c10::Stream) const noexcept override {
        return c10::Stream(c10::Stream::DEFAULT,
                           c10::Device(at::DeviceType::PrivateUse1, 0));
    }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }
    void record(void**, const c10::Stream&, const c10::DeviceIndex,
                const c10::EventFlag) const override {}
    void block(void*, const c10::Stream&) const override {}
    bool queryEvent(void*) const override { return true; }
    void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, NpuGuardImpl);

// ═══════════════════════════════════════════════════════════════════════════
// 3. Core ops: empty + copy_ (minimum required for device to work)
// ═══════════════════════════════════════════════════════════════════════════

at::Tensor npu_empty(c10::IntArrayRef size,
                     std::optional<at::ScalarType> dtype,
                     std::optional<at::Layout> layout,
                     std::optional<at::Device> device,
                     std::optional<bool> pin_memory,
                     std::optional<at::MemoryFormat> memory_format) {
    (void)layout; (void)device; (void)pin_memory;
    auto dt = dtype.value_or(at::ScalarType::Float);
    auto nbytes = c10::multiply_integers(size) * c10::elementSize(dt);

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(), nbytes,
        g_npu_allocator.allocate(nbytes), &g_npu_allocator, true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dt));
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    if (memory_format.has_value() &&
        *memory_format != at::MemoryFormat::Contiguous) {
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format);
    }
    return tensor;
}

at::Tensor& npu_copy_(at::Tensor& self, const at::Tensor& src,
                       bool non_blocking) {
    (void)non_blocking;
    // If same dtype and size, direct memcpy
    if (self.scalar_type() == src.scalar_type() && self.nbytes() == src.nbytes()) {
        if (self.data_ptr() != src.data_ptr()) {
            std::memcpy(self.data_ptr(), src.data_ptr(), self.nbytes());
        }
    } else {
        // Use CPU tensors for dtype conversion
        auto cpu_src = at::empty(src.sizes(),
            at::TensorOptions().dtype(src.scalar_type()).device(at::kCPU));
        std::memcpy(cpu_src.data_ptr(), src.data_ptr(), src.nbytes());
        auto cpu_dst = cpu_src.to(self.scalar_type());
        TORCH_CHECK(cpu_dst.nbytes() == self.nbytes(),
                    "copy_ size mismatch after cast: ", cpu_dst.nbytes(),
                    " vs ", self.nbytes());
        std::memcpy(self.data_ptr(), cpu_dst.data_ptr(), self.nbytes());
    }
    return self;
}

// aten::empty_strided — needed for .to() and many internal ops
at::Tensor npu_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                              std::optional<at::ScalarType> dtype,
                              std::optional<at::Layout> layout,
                              std::optional<at::Device> device,
                              std::optional<bool> pin_memory) {
    (void)layout; (void)device; (void)pin_memory;
    auto dt = dtype.value_or(at::ScalarType::Float);
    // Compute storage size from max offset
    int64_t storage_size = 1;
    for (size_t i = 0; i < size.size(); ++i) {
        if (size[i] > 0) {
            storage_size = std::max(storage_size,
                (size[i] - 1) * stride[i] + 1);
        }
    }
    auto nbytes = storage_size * static_cast<int64_t>(c10::elementSize(dt));

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(), nbytes,
        g_npu_allocator.allocate(nbytes), &g_npu_allocator, true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dt));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    return tensor;
}

// aten::zero_ — fill tensor with zeros
at::Tensor& npu_zero_(at::Tensor& self) {
    if (self.nbytes() > 0) {
        std::memset(self.data_ptr(), 0, self.nbytes());
    }
    return self;
}

// aten::fill_.Scalar — fill tensor with a scalar value
at::Tensor& npu_fill_scalar(at::Tensor& self, const at::Scalar& value) {
    auto dt = self.scalar_type();
    auto n = self.numel();
    if (n == 0) return self;

    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, dt, "npu_fill_", [&] {
        auto val = value.to<scalar_t>();
        auto* ptr = static_cast<scalar_t*>(self.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
            ptr[i] = val;
        }
    });
    return self;
}

// _copy_from and _copy_from_and_resize — needed by cpu_fallback
at::Tensor npu_copy_from(const at::Tensor& self, const at::Tensor& dst,
                          bool non_blocking) {
    (void)non_blocking;
    TORCH_CHECK(self.nbytes() == dst.nbytes(),
                "_copy_from size mismatch: ", self.nbytes(), " vs ", dst.nbytes());
    if (self.data_ptr() != dst.data_ptr()) {
        std::memcpy(dst.data_ptr(), self.data_ptr(), self.nbytes());
    }
    return dst;
}

at::Tensor npu_copy_from_and_resize(const at::Tensor& self,
                                     const at::Tensor& dst) {
    // Resize dst storage and metadata to match self
    auto required_bytes = self.nbytes();
    auto* impl = dst.unsafeGetTensorImpl();
    if (dst.nbytes() < required_bytes) {
        // Need to reallocate storage
        auto new_storage = c10::Storage(
            c10::Storage::use_byte_size_t(),
            required_bytes,
            &g_npu_allocator,
            /*resizable=*/true);
        impl->set_storage_and_dtype(new_storage, c10::scalarTypeToTypeMeta(self.scalar_type()));
    }
    impl->set_sizes_and_strides(self.sizes(), self.strides());
    if (required_bytes > 0 && self.data_ptr() != dst.data_ptr()) {
        std::memcpy(dst.data_ptr(), self.data_ptr(), required_bytes);
    }
    return dst;
}

// view — our storage is CPU memory so views work naturally
at::Tensor npu_view(const at::Tensor& self, c10::SymIntArrayRef size) {
    // Infer -1 dimensions
    std::vector<int64_t> inferred;
    inferred.reserve(size.size());
    int64_t neg_one_idx = -1;
    int64_t product = 1;
    for (size_t i = 0; i < size.size(); ++i) {
        auto s = size[i].expect_int();
        if (s == -1) {
            TORCH_CHECK(neg_one_idx == -1, "only one dimension can be inferred");
            neg_one_idx = static_cast<int64_t>(i);
            inferred.push_back(-1);
        } else {
            product *= s;
            inferred.push_back(s);
        }
    }
    if (neg_one_idx >= 0) {
        inferred[neg_one_idx] = self.numel() / product;
    }
    auto strides = at::detail::defaultStrides(inferred);
    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()), self.key_set(), self.dtype());
    result.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
    result.unsafeGetTensorImpl()->set_sizes_and_strides(inferred, strides);
    return result;
}

// as_strided — another view op, safe since storage is CPU memory
at::Tensor npu_as_strided(const at::Tensor& self, c10::SymIntArrayRef size,
                           c10::SymIntArrayRef stride,
                           std::optional<c10::SymInt> storage_offset) {
    std::vector<int64_t> sizes_vec, strides_vec;
    for (const auto& s : size) sizes_vec.push_back(s.expect_int());
    for (const auto& s : stride) strides_vec.push_back(s.expect_int());
    int64_t offset = storage_offset.has_value()
        ? storage_offset->expect_int() : self.storage_offset();
    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()), self.key_set(), self.dtype());
    result.unsafeGetTensorImpl()->set_storage_offset(offset);
    result.unsafeGetTensorImpl()->set_sizes_and_strides(sizes_vec, strides_vec);
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &npu_empty);
    m.impl("empty_strided", &npu_empty_strided);
    m.impl("copy_", &npu_copy_);
    m.impl("zero_", &npu_zero_);
    m.impl("fill_.Scalar", &npu_fill_scalar);
    m.impl("_copy_from", &npu_copy_from);
    m.impl("_copy_from_and_resize", &npu_copy_from_and_resize);
    m.impl("view", &npu_view);
    m.impl("as_strided", &npu_as_strided);
}

// _to_copy: explicit implementation to avoid cpu_fallback issues
at::Tensor npu_to_copy(const at::Tensor& self,
                        std::optional<at::ScalarType> dtype,
                        std::optional<at::Layout> layout,
                        std::optional<at::Device> device,
                        std::optional<bool> pin_memory,
                        bool non_blocking,
                        std::optional<at::MemoryFormat> memory_format) {
    (void)layout; (void)pin_memory; (void)non_blocking; (void)memory_format;
    auto target_device = device.value_or(self.device());
    auto target_dtype = dtype.value_or(self.scalar_type());

    if (target_device.type() == at::DeviceType::PrivateUse1) {
        // Creating NPU tensor
        auto result = npu_empty(self.sizes(), target_dtype,
                                 self.layout(), target_device,
                                 false, c10::nullopt);
        // If need dtype conversion, use CPU intermediate
        if (target_dtype != self.scalar_type()) {
            auto cpu_src = at::empty(self.sizes(),
                at::TensorOptions().dtype(self.scalar_type()).device(at::kCPU));
            std::memcpy(cpu_src.data_ptr(), self.data_ptr(), self.nbytes());
            auto cpu_converted = cpu_src.to(target_dtype);
            std::memcpy(result.data_ptr(), cpu_converted.data_ptr(), result.nbytes());
        } else {
            if (self.nbytes() > 0) {
                std::memcpy(result.data_ptr(), self.data_ptr(), self.nbytes());
            }
        }
        return result;
    } else {
        // Moving to CPU or other device
        auto cpu_result = at::empty(self.sizes(),
            at::TensorOptions().dtype(target_dtype).device(at::kCPU));
        if (target_dtype != self.scalar_type()) {
            auto cpu_src = at::empty(self.sizes(),
                at::TensorOptions().dtype(self.scalar_type()).device(at::kCPU));
            std::memcpy(cpu_src.data_ptr(), self.data_ptr(), self.nbytes());
            auto cpu_converted = cpu_src.to(target_dtype);
            std::memcpy(cpu_result.data_ptr(), cpu_converted.data_ptr(), cpu_result.nbytes());
        } else {
            if (self.nbytes() > 0) {
                std::memcpy(cpu_result.data_ptr(), self.data_ptr(), self.nbytes());
            }
        }
        if (target_device.type() == at::kCPU) {
            return cpu_result;
        }
        return cpu_result.to(target_device);
    }
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m2) {
    m2.impl("_to_copy", &npu_to_copy);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3b. Generic fallback is registered in eager_ops.cpp (OV dispatch + CPU fallback)
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// 4. Python bindings
// ═══════════════════════════════════════════════════════════════════════════

// From eager_ops.cpp
extern void register_eager_ops_bindings(py::module& m);

PYBIND11_MODULE(npu_backend, m) {
    m.doc() = "OpenVINO NPU backend (PrivateUse1) — device + OV eager ops";
    m.def("device_count", []() -> int64_t { return 1; });
    m.def("is_available", []() -> bool { return true; });
    m.def("current_device", []() -> int64_t { return 0; });
    c10::SetAllocator(at::DeviceType::PrivateUse1, &g_npu_allocator);
    c10::register_privateuse1_backend("npu");

    // OV eager ops stats
    register_eager_ops_bindings(m);
}
