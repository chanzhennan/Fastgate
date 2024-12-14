import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [os.path.join('backend', f'pybind.cpp'), 
           os.path.join('backend', f'hgemm.cu'),
        #    os.path.join('backend', f'hgemm_tr.cu'),
           os.path.join('backend', f'edgemm.cu'),
           os.path.join('backend', f'flatgemm.cu'),
        #    os.path.join('backend', f'edgemm_tr.cu'),
        #    os.path.join('backend', f'fastgemv.cu'),
           os.path.join('backend', f'fastgemv_czn.cu')]
            

# 自定义build_ext命令以支持多线程编译
class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        # 获取可用的CPU核心数量
        num_cores = multiprocessing.cpu_count()

        # 设置编译器选项，启用多线程编译
        for ext in self.extensions:
            ext.extra_compile_args = ['-j', str(num_cores)]  # 使用-j选项设置线程数
        super().build_extensions()

setup(
    name='eed',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('eed.backend',
            sources=sources,
            extra_compile_args = {
                'cxx': ['-O3', '-fopenmp', '-lgomp'],
                'nvcc': ['-O3',]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })