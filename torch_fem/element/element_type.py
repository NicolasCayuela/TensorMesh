import os
import re
import toml 
from .element import Element,\
                     Line,\
                     Triangle, \
                     Quadrilateral, \
                     Tetrahedron, \
                     Hexahedron, \
                     Prism, \
                     Pyramid
from typing import Callable,Dict,List,Type

with open(os.path.join(os.path.dirname(__file__), "dimension.toml"), "r") as f:
    element_type2dimension:Dict[str,int] = toml.load(f)
with open(os.path.join(os.path.dirname(__file__), "order.toml"), "r") as f:
    element_type2order:Dict[str,int] = toml.load(f)

element_types:List[str] = list(element_type2dimension.keys())

def element_type2element(x:str)->Type[Element]:
    element_prefix = re.findall(r'[a-zA-Z]+', x)[0]
    return {
        'line' : Line,
        'triangle' : Triangle, 
        'quad' : Quadrilateral,
        'tetra' : Tetrahedron,
        'hexahedron' : Hexahedron,
        'pyramid' : Pyramid,
        'wedge' : Prism
    }[element_prefix]

