from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cusparse_spmm',
    ext_modules=[
        CUDAExtension(
            'cusparse_spmm',
            ['cusparse_spmm.cpp'],
            extra_compile_args=['-O3']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
