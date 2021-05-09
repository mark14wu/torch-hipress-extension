from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from torch.utils import cpp_extension

setup(
    name='hipress-torch',
    ext_modules=[
        CUDAExtension('hp_cuda', [
            'hp_cuda.cpp',
            'hp_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

