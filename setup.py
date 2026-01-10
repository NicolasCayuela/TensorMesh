import os 
import re
import sys
from setuptools import setup, find_packages
import subprocess

# Function to run the setup script of a submodule
def build_submodule(submodule_path):
    subprocess.check_call(['python', 'setup.py', 'build_ext', '--inplace'], cwd=submodule_path)

# Only build C++ extension if torch is available and not in egg_info/metadata mode
# This allows pip to install dependencies first
def try_build_cpp_extension():
    # Skip build during pip's dependency resolution phase
    if any(arg in sys.argv for arg in ['egg_info', '--version', 'sdist']):
        return
    try:
        import torch
        build_submodule('tensormesh/cpp/spsolve')
    except ImportError:
        print("Warning: PyTorch not found. C++ sparse solver extension will not be built.")
        print("         The package will still work using the Python fallback solver.")
        print("         To enable the C++ backend, install PyTorch first, then reinstall tensormesh.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build C++ extension: {e}")
        print("         The package will still work using the Python fallback solver.")

try_build_cpp_extension()


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
        # "functorch"  # Removed: PyTorch 2.0+ has functorch built-in
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
