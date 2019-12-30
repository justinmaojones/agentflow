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
            'agentflow.buffers.prefix_sum_tree_methods',
            sources=[
                'agentflow/buffers/_prefix_sum_tree.pyx',
                'agentflow/buffers/prefix_sum_tree.cpp'],
            include_dirs=[numpy.get_include()],
            language='C++'),
    ]
 )
