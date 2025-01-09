import os
import glob
import multiprocessing
from setuptools import setup
import torch

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
    CUDA_HOME,
)

try:
    from torch_xmlir.utils.cpp_extension import XPUExtension, BuildExtension
    IS_KUNLUN_EXTENSION = True
except ImportError:
    IS_KUNLUN_EXTENSION = False


import subprocess
from typing import Set
import warnings

from packaging.version import parse, Version

from setuptools.command.install import install
import shutil

name = "speedgate"
current_directory = os.path.abspath(os.path.dirname(__file__))

class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        super().build_extensions()
    

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        print("Running custom install command...")

        # try:
        #     shutil.rmtree('build')
        #     print('Deleted build directory.')
        # except Exception as e:
        #     print(f"Error deleting build directory: {e}")

        # try:
        #     shutil.rmtree(f'{name}.egg-info')
        #     print(f'Deleted {name}.egg-info directory.')
        # except Exception as e:
        #     print(f"Error deleting {name}.egg-info directory: {e}")


ext_modules = []


def build_for_cuda():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

    # Supported NVIDIA GPU architectures.
    SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}

    # Compiler flags.
    Qserve_CXX_FLAGS = [
        "-g",
        "-O3",
        "-fopenmp",
        "-lgomp",
        "-std=c++17",
        "-DENABLE_BF16",
        "-DBUILD_WITH_CUDA",
    ]

    Qserve_NVCC_FLAGS = [
        "-O2",
        "-std=c++17",
        "-DENABLE_BF16",
        "-DBUILD_WITH_CUDA",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ]

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    Qserve_CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    Qserve_NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available to build the package.")

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.

        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    def get_torch_arch_list() -> Set[str]:
        # TORCH_CUDA_ARCH_LIST can have one or more architectures,
        # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
        # compiler to additionally include PTX code that can be runtime-compiled
        # and executed on the 8.6 or newer architectures. While the PTX code will
        # not give the best performance on the newer architectures, it provides
        # forward compatibility.
        env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if env_arch_list is None:
            return set()

        # List are separated by ; or space.
        torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
        if not torch_arch_list:
            return set()

        # Filter out the invalid architectures and print a warning.
        valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
        arch_list = torch_arch_list.intersection(valid_archs)
        # If none of the specified architectures are valid, raise an error.
        if not arch_list:
            raise RuntimeError(
                "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
                f"variable ({env_arch_list}) is supported. "
                f"Supported CUDA architectures are: {valid_archs}."
            )
        invalid_arch_list = torch_arch_list - valid_archs
        if invalid_arch_list:
            warnings.warn(
                f"Unsupported CUDA architectures ({invalid_arch_list}) are "
                "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
                f"({env_arch_list}). Supported CUDA architectures are: "
                f"{valid_archs}."
            )
        return arch_list

    # First, check the TORCH_CUDA_ARCH_LIST environment variable.
    compute_capabilities = get_torch_arch_list()
    if not compute_capabilities:
        # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
        # GPUs on the current machine.
        device_count = torch.cuda.device_count()
        gpu_version = None
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if gpu_version is None:
                gpu_version = f"{major}{minor}"
            else:
                if gpu_version != f"{major}{minor}":
                    raise RuntimeError(
                        "Kernels for GPUs with different compute capabilities cannot be installed simultaneously right now.\nPlease use CUDA_VISIBLE_DEVICES to specify the GPU for installation."
                    )
            if major < 7:
                raise RuntimeError("GPUs with compute capability below 7.0 are not supported.")
            compute_capabilities.add(f"{major}.{minor}")
    else:
        if len(compute_capabilities) > 1:
            raise RuntimeError(
                "Kernels for GPUs with different compute capabilities cannot be installed simultaneously right now.\nPlease restrict the length of TORCH_CUDA_ARCH_LIST to 1."
            )
        else:
            gpu_version = compute_capabilities[0].replace(".", "")

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if not compute_capabilities:
        # If no GPU is specified nor available, add all supported architectures
        # based on the NVCC CUDA version.
        compute_capabilities = SUPPORTED_ARCHS.copy()
        if nvcc_cuda_version < Version("11.1"):
            compute_capabilities.remove("8.6")
        if nvcc_cuda_version < Version("11.8"):
            compute_capabilities.remove("8.9")
            compute_capabilities.remove("9.0")

    # Validate the NVCC CUDA version.
    if nvcc_cuda_version < Version("11.0"):
        raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
    if nvcc_cuda_version < Version("11.1"):
        if any(cc.startswith("8.6") for cc in compute_capabilities):
            raise RuntimeError("CUDA 11.1 or higher is required for compute capability 8.6.")
    if nvcc_cuda_version < Version("11.8"):
        if any(cc.startswith("8.9") for cc in compute_capabilities):
            # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
            # However, GPUs with compute capability 8.9 can also run the code generated by
            # the previous versions of CUDA 11 and targeting compute capability 8.0.
            # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
            # instead of 8.9.
            warnings.warn(
                "CUDA 11.8 or higher is required for compute capability 8.9. "
                "Targeting compute capability 8.0 instead."
            )
            compute_capabilities = set(cc for cc in compute_capabilities if not cc.startswith("8.9"))
            compute_capabilities.add("8.0+PTX")
        if any(cc.startswith("9.0") for cc in compute_capabilities):
            raise RuntimeError("CUDA 11.8 or higher is required for compute capability 9.0.")

    # Add target compute capabilities to NVCC flags.
    for capability in compute_capabilities:
        num = capability[0] + capability[2]
        Qserve_NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            Qserve_NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    # Use NVCC threads to parallelize the build.
    if nvcc_cuda_version >= Version("11.2"):
        num_threads = min(os.cpu_count(), 8)
        Qserve_NVCC_FLAGS += ["--threads", str(num_threads)]

    cutlass_include_dir = os.path.join(current_directory, "3rdparty", "flash-attention", "csrc", "cutlass", "include")
    cutlass_tools_include_dir = os.path.join(
        current_directory,
        "3rdparty",
        "flash-attention",
        "csrc",
        "cutlass",
        "tools",
        "util",
        "include",
    )
    flash_attention_include_dir = os.path.join(
        current_directory, "3rdparty", "flash-attention", "csrc", "flash_attn", "src"
    )
    flash_attention_sources = glob.glob(
        os.path.join(
            current_directory,
            "3rdparty",
            "flash-attention",
            "csrc",
            "flash_attn",
            "src",
            "flash_fwd*.cu",
        )
    )
    flash_infer_include_dir = [
        os.path.join(current_directory, "3rdparty", "flashinfer", "include"),
    ]

    sources = (
        [
            os.path.join("inficom", "pybind.cpp"),
            os.path.join(
                "inficom", "kernels", "nvidia", "flash_attn", f"flash_api.cpp"
            ),
            os.path.join(
                "inficom", "kernels", "nvidia", "flash_attn", f"flash_attn.cpp"
            ),
            os.path.join(
                "inficom", "kernels", "nvidia", "flash_infer", f"flash_infer.cu"
            ),
            os.path.join("inficom", "kernels", "nvidia", "cutlass", "w8a8", f"w8a8.cu"),
            os.path.join(
                "inficom", "kernels", "nvidia", "cutlass", "dual_gemm", f"dual_gemm.cu"
            ),
            os.path.join("inficom", "kernels", "nvidia", "gemm_s4_f16", f"format.cu"),
            os.path.join(
                "inficom", "kernels", "nvidia", "gemm_s4_f16", f"gemm_s4_f16.cu"
            ),
            os.path.join(
                "inficom", "kernels", "nvidia", "marlin", f"marlin_cuda_kernel.cu"
            ),
            os.path.join("inficom", "ops", "unittest", f"ut.cpp"),
            os.path.join("inficom", "ops", "nvidia", "attn", f"attention.cu"),
            os.path.join("inficom", "ops", "nvidia", "norm", f"norm.cu"),
            os.path.join("inficom", "ops", "nvidia", "embedding", f"embedding.cu"),
            os.path.join("inficom", "ops", "nvidia", "linear", f"gemm.cu"),
            os.path.join("inficom", "ops", "nvidia", "linear", f"gemm_w4a16.cu"),
            os.path.join("inficom", "ops", "nvidia", "linear", f"gemm_w8a8.cu"),
            os.path.join("inficom", "ops", "nvidia", "element", f"residual.cu"),
            os.path.join("inficom", "ops", "nvidia", "element", f"activation.cu"),
            os.path.join("inficom", "ops", "nvidia", "cache", f"cache.cu"),
            os.path.join("inficom", "layers", f"attn_layer.cpp"),
            os.path.join("inficom", "layers", f"attn_layer_long.cpp"),
            os.path.join("inficom", "layers", f"ffn_layer.cpp"),
            os.path.join("inficom", "layers", f"ffn_layer_long.cpp"),
        ]
        + flash_attention_sources
        + []
    )

    qgemm_w4a8_per_chn_extension = CUDAExtension(
        name="inficom_w4a8_per_chn",
        sources=[
            "inficom/ops/nvidia/qserve/w4a8_per_chn/gemm_cuda.cu",
            "inficom/ops/nvidia/qserve/w4a8_per_chn/pybind.cpp",
        ],
        extra_objects=[f"inficom/kernels/nvidia/qserve/w4a8_per_chn/libs/gemm_cuda_kernel_sm_{gpu_version}.so"],
        extra_compile_args={
            "cxx": Qserve_CXX_FLAGS,
            "nvcc": Qserve_NVCC_FLAGS,
        },
    )
    ext_modules.append(qgemm_w4a8_per_chn_extension)

    qgemm_w4a8_per_group_extension = CUDAExtension(
        name="inficom_w4a8_per_group",
        sources=[
            "inficom/ops/nvidia/qserve/w4a8_per_group/gemm_cuda.cu",
            "inficom/ops/nvidia/qserve/w4a8_per_group/pybind.cpp",
        ],
        extra_objects=[f"inficom/kernels/nvidia/qserve/w4a8_per_group/libs/gemm_cuda_kernel_sm_{gpu_version}.so"],
        extra_compile_args={
            "cxx": Qserve_CXX_FLAGS,
            "nvcc": Qserve_NVCC_FLAGS,
        },
    )
    ext_modules.append(qgemm_w4a8_per_group_extension)

    qgemm_w8a8_extension = CUDAExtension(
        name="inficom_w8a8",
        sources=[
            "inficom/ops/nvidia/qserve/w8a8/gemm_cuda.cu",
            "inficom/ops/nvidia/qserve/w8a8/pybind.cpp",
        ],
        extra_objects=[f"inficom/kernels/nvidia/qserve/w8a8/libs/gemm_cuda_kernel_sm_{gpu_version}.so"],
        extra_compile_args={
            "cxx": Qserve_CXX_FLAGS,
            "nvcc": Qserve_NVCC_FLAGS,
        },
    )
    ext_modules.append(qgemm_w8a8_extension)

    # Fuse kernels.
    fused_extension = CUDAExtension(
        name="inficom_fused_kernels",
        sources=[
            "inficom/ops/nvidia/qserve/fused/fused.cpp",
            "inficom/ops/nvidia/qserve/fused/fused_kernels.cu",
        ],
        extra_compile_args={
            "cxx": Qserve_CXX_FLAGS,
            "nvcc": Qserve_NVCC_FLAGS,
        },
    )
    ext_modules.append(fused_extension)

    # attention from fastertransformer
    fused_attention_extension = CUDAExtension(
        name="inficom_fused_attention",
        sources=[
            "inficom/ops/nvidia/qserve/fused_attention/fused_attention.cpp",
            "inficom/ops/nvidia/qserve/fused_attention/decoderMaskedMultiheadAttention.cu",
            "inficom/ops/nvidia/qserve/fused_attention/update_kv_cache.cu",
            "inficom/ops/nvidia/qserve/fused_attention/input_metadata_helper.cu",
        ],
        extra_compile_args={
            "cxx": Qserve_CXX_FLAGS,
            "nvcc": Qserve_NVCC_FLAGS,
        },
    )
    ext_modules.append(fused_attention_extension)

    compute_capability = torch.cuda.get_device_capability()
    cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

    ext_modules.append(
        CUDAExtension(
            name,
            sources=sources,
            include_dirs=[
                cutlass_include_dir,
                cutlass_tools_include_dir,
                flash_attention_include_dir,
                *flash_infer_include_dir,
            ],
            libraries=[
                "cublas",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-fopenmp",
                    "-lgomp",
                    "-DBUILD_WITH_CUDA",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-line-info",
                    "-Xptxas=-v",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    f"-DCUDA_ARCH={cuda_arch}",
                ],
            },
        ),
    )


def build_for_kunlun():

    sources = [
        os.path.join("backend", "kunlun", f"kernels.xpu"),
        os.path.join("backend", "kunlun", f"rope.xpu"),
        os.path.join("backend", "kunlun", f"lltm.cpp"),
        os.path.join("backend", "kunlun", f"utils.cpp"),
    ]

    ext_modules.append(
        XPUExtension(
            name,
            sources=sources,
            library_dirs=["/usr/local/xcudart/lib"],
            # 头文件路径
            include_dirs=["/ssd3/zhennanc/baidu/xpu/XMLIR/third_party/xhpc/xdnn/include",
            "/ssd3/zhennanc/baidu/xpu/XMLIR/third_party/xre/include",
            "/ssd3/zhennanc/baidu/xpu/XMLIR/xdnn_pytorch/include",
            "/ssd3/zhennanc/baidu/xpu/XMLIR/runtime/include",
            "/ssd3/zhennanc/baidu/xpu/XMLIR/third_party/xccl/include/"],
        ),
         
    )

BUILD_TARGET = os.environ.get("BUILD_TARGET", "kunlun")

if BUILD_TARGET == "auto":
        build_for_cuda()
else:
    if BUILD_TARGET == "cuda":
        build_for_cuda()
    elif BUILD_TARGET == "kunlun":
        build_for_kunlun()

setup(
    name=name,
    version="0.2.3",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": CustomBuildExt,
        "install": CustomInstallCommand,
    },
    verbose=True
)