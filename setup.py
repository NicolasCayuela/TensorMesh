import os 
import re
from setuptools import setup, find_packages
import subprocess

# Function to run the setup script of a submodule
def build_submodule(submodule_path):
    subprocess.check_call(['python', 'setup.py', 'build_ext', '--inplace'], cwd=submodule_path)

build_submodule('tensormesh/cpp/spsolve')


def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'tensormesh', '_version.py')
    with open(version_file, 'r') as f:
        version_content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="tensormesh",
    version=read_version(),
    author="walkerchi,shizhengwen",
    author_email="walker.chi.000@gmail.com",
    description="Differentialable Numerical Method for Partial Differential Equation Library for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://torch-fem.readthedocs.io",
    project_urls={
        "Documentation": "https://torch-fem.readthedocs.io",
        "Source": "https://github.com/walkerchi/torch-fem.git",
        "Changelog": "https://github.com/walkerchi/torch-fem/blob/master/CHANGELOG.md",
    },
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "torch>=1.8.0",
        "meshio",
        "matplotlib", 
        "psutil",
        "toml",
        "functorch; torch<2.0"
    ],
    extras_require={
        "petsc":[
            "petsc4py"
        ],
        "cupy":[
            "cupy"
        ],
        "example":[
            "plotly"
        ],
        "test": [
            "pytest",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[
        "deep-learning",
        "AI4S",
        "ai-for-science",
        "pytorch",
        "numerical",
        "partial-differential-equation",
        "finite-element-methods",
        "geometric-deep-learning",
        "graph-neural-networks",
        "graph-convolutional-networks",
    ],
    python_requires=">=3.8",
)
