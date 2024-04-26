# import torch 
# import sys
# sys.path.append("../..")
# from torch_fem.element import element_type2element, \
#                               element_type2order 
# from torch_fem import Mesh
# class NaiveElementAssembler:
#     def __init__(self, 
#                  points:torch.Tensor,
#                  elements:Dict[str,torch.Tensor]):
#         pass 

#     @classmethod 
#     def from_mesh(cls, mesh:Mesh):
#         mesh.elements()
#         if 