import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
CONDA_PREFIX = osp.abspath(os.environ.get('CONDA_PREFIX', ''))
CUDA_TARGET_INCLUDE = osp.join(CONDA_PREFIX, 'targets', 'x86_64-linux', 'include')

EIGEN_INCLUDE = osp.join(ROOT, 'thirdparty/eigen-3.4.0')
if not osp.isdir(EIGEN_INCLUDE):
    EIGEN_INCLUDE = osp.join(CONDA_PREFIX, 'include', 'eigen3')


setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            include_dirs=[CUDA_TARGET_INCLUDE],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                EIGEN_INCLUDE, CUDA_TARGET_INCLUDE]
            ),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'), 
                EIGEN_INCLUDE, CUDA_TARGET_INCLUDE],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp', 
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
