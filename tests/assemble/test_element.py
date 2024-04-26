# TODO
# add linear elasticity

import sys 
import numpy as np
import torch
import meshio
sys.path.append("../..")

from torch_fem import ElementAssembler, NodeAssembler,  Mesh
from torch_fem import dot, mul
import skfem

class LaplaceAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        K = dot(gradu, gradv)
        return K

class ProductAssembler(ElementAssembler):
    def forward(self, u, v):
        K = mul(u, v)
        
        return K

@skfem.BilinearForm
def laplace_assembler(u, v, w):
    from skfem.helpers import dot,grad 
    return dot(grad(u), grad(v))

@skfem.BilinearForm
def product_assembler(u, v, w):
    from skfem.helpers import dot,grad 
    return u * v

def element_assemble(mesh, model="laplace"):
    assert model in ["laplace", "product"]
    assambler = {
        "laplace": LaplaceAssembler,
        "product": ProductAssembler
    }[model]
    K_asm = assambler.from_mesh(mesh, quadrature_order=2)
    K = K_asm(mesh.points)
    
    K_scipy = K.to_scipy_coo()
    if mesh.default_element_type.startswith("tri"):
        Mesh = skfem.MeshTri
    elif mesh.default_element_type.startswith("quad"):
        Mesh = skfem.MeshQuad
    elif mesh.default_element_type.startswith("tet"):
        Mesh = skfem.MeshTet
    else:
        raise NotImplementedError
    # breakpoint()
    if mesh.default_element_type == "quad":
        elements = mesh.elements().T.numpy()
        elements[[2,3]] = elements[[3,2]]
        breakpoint()
    else:
        elements = mesh.elements().T.numpy()
    mesh_skfem = Mesh(mesh.points.T.cpu().numpy(), elements)

    skfem_assembler = {
        "laplace": laplace_assembler,
        "product": product_assembler
    }[model]

    element = {
        "triangle": skfem.ElementTriP1(),
        "quad": skfem.ElementQuad1(),
        "tetra": skfem.ElementTetP1(),
    }[mesh.default_element_type]
    basis = skfem.InteriorBasis(mesh_skfem, element)

    K_skfem = skfem.asm(skfem_assembler, basis)

    K_scipy_dense = K_scipy.toarray() # [n_node, n_node]
    K_skfem_dense = K_skfem.toarray()

    # breakpoint()
    np.testing.assert_allclose(K_scipy_dense, K_skfem_dense, rtol=1e-5)
    for K in [K_scipy_dense, K_skfem_dense]:
        min_lambda = np.linalg.eig(K)[0].min()
        if min_lambda < 0:
            np.testing.assert_allclose(min_lambda, 0, atol=1e-10)


def test_tri1_1():
    # element_assemble(Mesh.gen_rectangle(chara_length=0.2,  element_type="tri"), model="laplace")
    element_assemble(Mesh.gen_rectangle(chara_length=0.2,  element_type="quad"), model="product")

def test_tri1_2():
    m = skfem.MeshTri().refined(4)
    mesh = meshio.Mesh(points = m.p.T, cells = [("triangle", m.t.T)])
    mesh = Mesh.from_meshio(mesh)
    # element_assemble(mesh, model="laplace")
    element_assemble(mesh, model="product")

def test_quad1():
    element_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="quad"), model="laplace")
    element_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="quad"), model="product")

def test_tet1():
    element_assemble(Mesh.gen_cube(chara_length=0.1), model="laplace")
    element_assemble(Mesh.gen_cube(chara_length=0.1), model="product")


