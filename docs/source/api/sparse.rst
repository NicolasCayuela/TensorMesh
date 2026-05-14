tensormesh.sparse
=================

.. py:module:: tensormesh.sparse

Sparse matrix data type and solver entry points. Built on top of
``torch-sla``; see :doc:`/user_guide/linear_solvers` for the design and
backend matrix.

SparseMatrix
------------

.. currentmodule:: tensormesh.sparse

.. autoclass:: tensormesh.sparse.SparseMatrix
    :members:
    :show-inheritance:
    :exclude-members: layout_hash

.. py:class:: SparseTensor

   COO sparse tensor with autograd support — the base class that
   :class:`~tensormesh.sparse.SparseMatrix` extends. Re-exported from
   ``torch-sla``; see the
   `torch-sla docs <https://github.com/sparsexlab/torch-sla>`_ for the
   full API.


Solvers
-------

.. autofunction:: tensormesh.sparse.spsolve

.. autofunction:: tensormesh.sparse.spmm

.. autofunction:: tensormesh.sparse.nonlinear_solve


Backend availability flags
--------------------------

Module-level booleans, set at import time, indicating which **optional**
backends were detected. ``torch-sla`` itself is a hard dependency — its
absence raises at import time rather than flipping a flag — so it is not
listed here.

.. autodata:: tensormesh.sparse.is_cpp_backend_available
   :annotation:

.. autodata:: tensormesh.sparse.is_petsc_available
   :annotation:

.. autodata:: tensormesh.sparse.is_cupy_available
   :annotation:
