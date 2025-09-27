from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

def get_cuda_extensions():
    cuda_sources = glob.glob('src/backends/cuda/kernels/*.cu')
    cpp_sources = glob.glob('src/backends/cusparse/*.cpp')
    
    if not cuda_sources and not cpp_sources:
        return []
    
    extensions = []
    if cuda_sources:
        extensions.append(
            CUDAExtension(
                'graph_conv_cuda',
                cuda_sources + ['src/backends/cuda/bindings.cpp'],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math', '-arch=sm_80']
                }
            )
        )
    
    if cpp_sources:
        extensions.append(
            CUDAExtension(
                'graph_conv_cusparse',
                cpp_sources,
                extra_compile_args=['-O3'],
                libraries=['cusparse']
            )
        )
    
    return extensions

setup(
    name='graph_nn_benchmarks',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=get_cuda_extensions(),
    cmdclass={'build_ext': BuildExtension} if get_cuda_extensions() else {},
    python_requires='>=3.8',
    install_requires=[
        'torch==2.4.0',
        'numpy<2.0',
        'scipy>=1.7.0',
        'dgl>=2.4.0',
        'torch-geometric>=2.3.0',
        'ogb>=1.3.0',
        'triton>=2.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'pandas>=1.5.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'tensorboard>=2.10.0',
        'wandb>=0.15.0',
    ],
)
