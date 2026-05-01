Installation
============

TensorMesh runs on Linux, macOS, and Windows. The only hard requirements are
**Python ≥ 3.10** and **PyTorch ≥ 2.0**; everything else (NumPy, SciPy, meshio,
``torch-sla``, ...) is pulled in automatically by ``pip``.

If you plan to use the GPU backend, install a CUDA-enabled PyTorch build *before*
installing TensorMesh — follow the official `PyTorch installation selector
<https://pytorch.org/get-started/locally/>`_ for the right command on your
platform / CUDA version.


Install via PyPI
----------------

The recommended way to install TensorMesh is from PyPI:

.. code-block:: bash

    pip install tensor-mesh

This pulls in all required dependencies, including
`torch-sla <https://www.torchsla.com/>`_, the differentiable sparse linear
algebra library that powers TensorMesh's solvers.


Install from source
-------------------

For development work, or to get the latest unreleased changes, clone the
repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/camlab-ethz/TensorMesh.git
    cd TensorMesh
    pip install -e .

The ``-e`` (editable) flag means edits to the source tree are picked up
without reinstalling. To install with the test dependencies as well:

.. code-block:: bash

    pip install -e ".[test]"

.. note::

   The first install builds a small C++ extension for the sparse solver. If
   PyTorch is missing or the build fails, TensorMesh falls back to a pure
   Python solver and prints a warning — the package still works, just with
   reduced performance on large systems.


Optional extras
---------------

Several features depend on packages that are *not* installed by default. Pick
the extras you need:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - Install command
   * - PETSc solver backend
     - ``pip install "tensor-mesh[petsc]"``
   * - CuPy GPU sparse backend
     - ``pip install "tensor-mesh[cupy]"``
   * - Plotly for example notebooks
     - ``pip install "tensor-mesh[example]"``
   * - Test suite (pytest)
     - ``pip install "tensor-mesh[test]"``

Two further packages are commonly useful but are *not* declared as extras —
install them directly when needed:

.. code-block:: bash

    pip install gmsh       # external mesh generation / .msh I/O
    pip install pyvista    # interactive 3D visualization


Next steps
----------

Once installed, head to :doc:`verify_install` to run a smoke test, or jump
straight into :doc:`quickstart` for a 2D Poisson walkthrough.
