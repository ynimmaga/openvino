# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools.command.install import install

# OpenVINO paths
_ov_pkg = os.path.join(
    os.environ.get("VIRTUAL_ENV", ""),
    "lib", "python3.10", "site-packages", "openvino",
)
_ov_include = os.path.join(_ov_pkg, "include")
_ov_libs = os.path.join(_ov_pkg, "libs")

# PyTorch frontend public headers (for pytorch/frontend.hpp, pytorch/decoder.hpp)
_pt_fe_include = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "pytorch", "include",
)


class InstallWithPth(install):
    """After install, drop a .pth file so intel_npu auto-loads with Python."""
    def run(self):
        super().run()
        # Write .pth file next to site-packages
        import site
        sp = site.getsitepackages()[0] if site.getsitepackages() else self.install_lib
        pth = os.path.join(sp, "intel_npu.pth")
        with open(pth, "w") as f:
            f.write("import intel_npu\n")


setup(
    name="intel_npu",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="npu_backend",
            sources=[
                "csrc/npu_backend.cpp",
                "csrc/eager_ops.cpp",
            ],
            include_dirs=[_ov_include, _pt_fe_include],
            extra_compile_args=["-std=c++17"],
            extra_link_args=[
                os.path.join(_ov_libs, "libopenvino.so.2610"),
                os.path.join(_ov_libs, "libopenvino_pytorch_frontend.so.2610"),
                f"-Wl,-rpath,{_ov_libs}",
            ],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "install": InstallWithPth,
    },
    packages=["intel_npu"],
    package_dir={"intel_npu": "."},
)
