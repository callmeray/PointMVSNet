from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {'cxx': ['-g'],
                      'nvcc': ['-O2']}

setup(
    name='dgcnn_ext',
    ext_modules=[
        CUDAExtension(
            name='dgcnn_ext',
            sources=[
                'csrc/main.cpp',
                'csrc/gather_knn_kernel.cu',
            ],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
