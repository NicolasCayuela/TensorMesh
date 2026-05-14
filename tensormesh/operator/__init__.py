"""Boundary-condition operators.

Currently exposes :class:`Condenser`, which applies Dirichlet boundary
conditions to an assembled FEM system via static condensation. See the
class docstring and the User Guide chapter on boundary conditions for
the high-level walk-through.
"""

from .condense import Condenser

__all__ = ["Condenser"]
