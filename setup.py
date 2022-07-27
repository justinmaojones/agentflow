from setuptools import find_packages, setup, Extension

requirements = (open("requirements.txt").read(),)
try:
    requirements_dev = (open("requirements_dev.txt").read(),)
    extras_require = {"dev": requirements_dev}
except IOError:
    extras_require = None


def get_long_description():
    # TODO: fix this hacky method of resolving relative links in pypi readme viewer
    README = open("README.md").read()
    base_url = "https://github.com/justinmaojones/agentflow/blob/master/"
    relative_links = [
        "docs/badges/python.svg",
        "docs/badges/coverage.svg",
    ]
    for rlink in relative_links:
        abs_link = base_url + rlink
        README = README.replace(rlink, abs_link)
    return README


setup(
    name="agentflow",
    version="0.1",
    description="AgentFlow Reinforcement Learning Library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/justinmaojones/agentflow",
    author="Justin Mao-Jones",
    author_email="justinmaojones@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(include=["agentflow", "agentflow*"]),
)
