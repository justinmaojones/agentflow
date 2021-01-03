from distutils.core import setup, Extension

setup(
	name='agentflow',
	version='1.0',
	description='reinforcement learning project',
	author='Justin Mao-Jones',
	author_email='justinmaojones@gmail.com',
	install_requires=open('requirements.txt').read(),
	packages=['agentflow'],
 )
