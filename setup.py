from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy

setup(
	name='agentflow',
	version='1.0',
	description='reinforcement learning project',
	author='Justin Mao-Jones',
	author_email='justinmaojones@gmail.com',
	install_requires=open('requirements.txt').read(),
	packages=[],
    cmdclass={'build_ext': build_ext},
    ext_modules= [
        Extension(
            'agentflow.buffers.segment_tree_c',
            sources=['src/_segment_tree.pyx','src/segment_tree.cpp'],
            include_dirs=[numpy.get_include()],
            extra_compile_args = ["-Xpreprocessor","-fopenmp",'-lomp'],
            extra_link_args=['-Xpreprocessor','-fopenmp','-lomp'],
            language='C++'),
    ]
 )
