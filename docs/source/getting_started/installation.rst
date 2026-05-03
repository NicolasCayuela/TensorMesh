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
   :widths: 28 72

   * - Extra
     - Install command
   * - PETSc solver backend (``petsc4py``)
     - ``pip install "tensor-mesh[petsc]"``
   * - CuPy GPU sparse backend (``cupy``)
     - ``pip install "tensor-mesh[cupy]"``
   * - Plotly for example notebooks
     - ``pip install "tensor-mesh[example]"``
   * - Test suite (pytest, pytest-cov)
     - ``pip install "tensor-mesh[test]"``

Two further packages are commonly useful but are *not* declared as extras —
install them directly when needed:

.. code-block:: bash

    pip install gmsh       # external mesh generation / .msh I/O
    pip install pyvista    # interactive 3D visualization


Sparse solvers and GPU acceleration
-----------------------------------

The sparse linear-algebra layer that powers TensorMesh's solvers has been
split out into a standalone library,
`torch-sla <https://github.com/sparsexlab/torch-sla>`_, so that it can evolve
independently and serve other projects. ``torch-sla`` is a required
dependency of TensorMesh and ships all current and future solver
improvements; it is the canonical entry point for both CPU and GPU sparse
solves.

The original solver code under ``tensormesh.sparse.solve`` (covering
``scipy``, ``petsc``, ``cupy``, ``cudss``, ``amg``, and ``torch`` backends)
is still present and used as a **fallback** when ``torch-sla`` is not
available — :func:`~tensormesh.spsolve` will pick it up automatically. New
features and performance work, however, land in ``torch-sla`` first; for
production use, prefer the ``torch_sla`` backend.

To enable GPU sparse solves, install ``torch-sla`` with the ``[cuda]``
extra:

.. code-block:: bash

    pip install "torch-sla[cuda]"

This pulls in ``cupy-cuda12x`` (CUDA 12 wheels of CuPy) and
``nvmath-python`` (NVIDIA's cuDSS bindings), which together give
``torch-sla`` access to its CuPy and cuDSS GPU solvers.

The TensorMesh-side extras serve the in-tree fallback path:

* ``tensor-mesh[cupy]`` — installs the generic ``cupy`` package, enabling
  the fallback ``cupy`` and ``cudss`` backends when ``torch-sla`` is not
  installed or unavailable.
* ``tensor-mesh[petsc]`` — installs ``petsc4py`` for the direct PETSc
  backend. This is independent of ``torch-sla`` and is useful for very
  large problems or when interoperating with an existing PETSc
  installation.


Next steps
----------

Once installed, head to :doc:`verify_install` to run a smoke test, or jump
straight into :doc:`quickstart` for a 2D Poisson walkthrough.
