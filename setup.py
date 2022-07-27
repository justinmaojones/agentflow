from setuptools import find_packages, setup, Extension

setup(
	name='agentflow',
	version='0.1',
	description='AgentFlow Reinforcement Learning Library',
    url="https://github.com/justinmaojones/agentflow",
	author='Justin Mao-Jones',
	author_email='justinmaojones@gmail.com',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
	install_requires=open('requirements.txt').read(),
    packages=find_packages(include=["agentflow", "agentflow.*"]),
 )
