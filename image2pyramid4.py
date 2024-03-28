import os
import subprocess
from torch.utils.cpp_extension import CUDA_HOME
import torch
from torch.utils.cpp_extension import load
# ------------------------------------------------------------------------------------
#                                   jit load 
# ------------------------------------------------------------------------------------

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []

cc_flag.append("-gencode=arch=compute_80,code=sm_80")
# cc_flag.append("arch=compute_80,code=sm_80")

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

this_dir = f'{os.path.dirname(os.path.abspath(__file__))}'
image2pyramid4_cuda = load(
    name="image2pyramid4", 
    sources=[str(this_dir) + "/image2pyramid4.cpp", 
                str(this_dir)+'/image2pyramid4_kernel.cu'], 
    # extra_include_paths=[str(this_dir) + '/cub/'],
    extra_cflags = ["-O3", "-std=c++17"] + generator_flag,
    extra_cuda_cflags = append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    # "-g",
                    # "-G",
                ]
                + generator_flag
                + cc_flag
            ),
    with_cuda=True)





def avgpool_pyramid4_forward(
	x:torch.Tensor):
	return image2pyramid4_cuda.avgpool_pyramid4_forward(x)


def cross_accum_forward(
	x:torch.Tensor):
	return image2pyramid4_cuda.cross_accum_forward(x)
