from setuptools import find_packages, setup


def _fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


def _fetch_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="prefpalette",
    version="0.1.0",
    packages=find_packages(exclude=("data", "docs", "examples", "configs")),
    description="PrefPalette: Personalized Preference Modeling with Latent Attributes",
    long_description=_fetch_readme(),
    long_description_content_type="text/markdown",
    install_requires=_fetch_requirements("requirements.txt"),
    python_requires=">=3.10",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
)
