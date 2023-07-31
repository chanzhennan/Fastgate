import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [os.path.join('backend', f'pybind.cpp'), 
           os.path.join('backend', f'hgemm.cu'),
           os.path.join('backend', f'edgemm.cu')]

setup(
    name='eed',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('eed.backend',
            sources=sources,
            extra_compile_args = {
                'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
                'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })