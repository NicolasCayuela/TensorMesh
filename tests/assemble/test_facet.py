
import sys 
import numpy as np
import torch
import meshio
sys.path.append("../..")

from torch_fem import ElementAssembler, NodeAssembler, FacetAssembler, Mesh
from torch_fem import dot, mul
import skfem

class ProductAssembler(FacetAssembler):
    def forward(self, u, v):
        return u * v
    
class LaplaceAssembler(FacetAssembler):
    def forward(self, gradu, gradv):
        return dot(gradu, gradv)
    

def test_facet_shape():
    mesh = Mesh.gen_rectangle(0.1)
    assembler = ProductAssembler.from_mesh(mesh, quadrature_order=2)
    V = assembler()

    
