# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools.command.install import install

_here = os.path.dirname(os.path.abspath(__file__))

# OpenVINO runtime location.
#
# By default we use an in-tree build installed under <repo>/dist (produced by
# `cmake --install build --prefix dist`; this file lives at src/frontends/eager,
# so the repo root is three levels up). Override with OPENVINO_DIST to point at
# any other OpenVINO runtime root (e.g. a pip-installed openvino package).
_repo_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))
_ov_dist = os.environ.get(
    "OPENVINO_DIST", os.path.join(_repo_root, "dist", "runtime")
)
_ov_include = os.path.join(_ov_dist, "include")
_ov_libs = os.path.join(_ov_dist, "lib", "intel64")


def _ov_lib(stem):
    """Resolve the unversioned .so symlink for an OpenVINO library.

    Falls back to the highest-versioned .so.* if the bare symlink is absent,
    so the build does not hard-code a soversion (the in-tree build is 2620 /
    2026.2.0, but this stays correct as OpenVINO is bumped)."""
    bare = os.path.join(_ov_libs, stem + ".so")
    if os.path.exists(bare):
        return bare
    candidates = sorted(glob.glob(os.path.join(_ov_libs, stem + ".so.*")))
    if not candidates:
        raise RuntimeError(
            f"Could not find {stem}.so under {_ov_libs}. "
            "Build and install OpenVINO first, or set OPENVINO_DIST to a "
            "runtime that contains it."
        )
    return candidates[-1]


# PyTorch frontend public headers (for pytorch/frontend.hpp, pytorch/decoder.hpp).
# These live in the sibling pytorch frontend, not under eager/.
_pt_fe_include = os.path.join(_here, "..", "pytorch", "include")


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
                "common/src/npu_backend.cpp",
                "pytorch/src/eager_ops.cpp",
            ],
            include_dirs=[_ov_include, _pt_fe_include, "pytorch/src"],
            extra_compile_args=["-std=c++17"],
            extra_link_args=[
                _ov_lib("libopenvino"),
                _ov_lib("libopenvino_pytorch_frontend"),
                f"-Wl,-rpath,{_ov_libs}",
            ],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "install": InstallWithPth,
    },
    packages=["intel_npu"],
)
