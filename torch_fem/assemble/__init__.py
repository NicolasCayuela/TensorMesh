from .assembler import ElementAssembler, NodeAssembler

from ..utils import dot, mul, sym, ddot, eye, trace


class LaplaceElementAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        K = dot(gradu, gradv)
        return K
    
class MassElementAssembler(ElementAssembler):
    def forward(self, u, v):
        K = mul(u, v)
        return K