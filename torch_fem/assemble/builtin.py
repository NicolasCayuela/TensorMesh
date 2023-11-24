from .element_assembler import ElementAssembler
from .node_assembler import NodeAssembler

from ..functional import dot, mul, sym, ddot, eye, trace


class LaplaceElementAssembler(ElementAssembler):
    r"""The element laplace assembler

    .. math::
    
        K = \int_{\Omega}\nabla u \cdot \nabla v \mathrm{d}v
    
    """
    def forward(self, gradu, gradv):
        K = dot(gradu, gradv)
        return K
    
class MassElementAssembler(ElementAssembler):
    r"""The element mass assembler
    
    .. math::
        
        K = \int_{\Omega} u v \mathrm{d}v
        
    """
    def forward(self, u, v):
        K = mul(u, v)
        return K
    
class ConstNodeAssembler(NodeAssembler):
    r"""The const node assembler
    
    .. math::

        f = \int_{\Omega} c\cdot u \mathrm{d}v
    
    """
    def __post_init__(self, c=1):
        self.c = c
    def forward(self, u, v):
        f = self.c * u
        return f
    
__all__ = ["LaplaceElementAssembler", "MassElementAssembler", "ConstNodeAssembler"]