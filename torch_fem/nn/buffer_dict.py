import re
import torch
import torch.nn as nn
from itertools import chain
from collections import OrderedDict
from typing import Union, Sequence, Optional,Dict,Iterable, Tuple, Mapping


class BufferDict(nn.Module):
    def __init__(self, data:Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        if data is None:
            data= {}
        self._data:Dict[str,torch.Tensor] = OrderedDict() # used for storing data that cannot be used as a valid name
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
      
        for key in list(data.keys()):
            if not pattern.match(key):
                self._data[key] = data.pop(key)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")
    
    def as_parameter(self, key:str):
        """Convert a buffer to a parameter"""
        buffer = self._buffers.pop(key) 
        self.register_parameter(key, buffer) # type: ignore
        
    def as_buffer(self, key:str):
        """Convert a parameter to a buffer"""
        parameter = self._parameters.pop(key)
        self.register_buffer(key, parameter)
        
    def keys(self)->Iterable[str]:
        return chain(self._buffers.keys(), self._parameters.keys(), self._data.keys())
    
    def items(self)->Iterable[Tuple[str, torch.Tensor]]:
        return chain(self._buffers.items(), self._parameters.items(), self._data.items()) # type: ignore
    
    def values(self)->Iterable[torch.Tensor]:
        return chain(self._buffers.values(), self._parameters.values(), self._data.values()) # type: ignore
    
    def __hash__(self):
        return hash((super().__hash__(), hash(tuple(self._data.keys())), hash(tuple(self._data.values()))))

    def __getitem__(self, key:str)->torch.Tensor:
        if key not in self.keys():
            raise KeyError(f"{key} is not found in the BufferDict")
        
        assert key in self.keys()

        if key in self._buffers.keys():
            return self._buffers[key] # type: ignore
        elif key in self._parameters.keys():
            return self._parameters[key] # type: ignore
        else:
            return self._data[key]
        
    def __setitem__(self, key:str, value:torch.Tensor):
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
        if not pattern.match(key):
            self._data[key] = value
        else:
            if isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")

    def __len__(self):
        return len(self._buffers) + len(self._parameters) + len(self._data)
    
    def __includes__(self, key):
        return key in self.keys()

    def is_floating_point(self)->bool:
        return any(map(lambda x:x.is_floating_point(), self.values()))

    def is_complex(self)->bool:
        return any(map(lambda x:x.is_complex(), self.values()))

    @property
    def dtype(self):
        return next(iter(self.buffers().values())).dtype # type: ignore

    @property
    def device(self):
        return next(iter(self.buffers().values())).device # type: ignore
    
    def _apply(self, fn):
        self = super()._apply(fn)
        for key, value in self._data.items():
            self._data[key] = fn(value)
        return self

    def __str__(self):
        return f"""BufferDict(
        {', '.join([f"{key} = {value}" for key, value in self.items()])}
        )"""
      
    def __repr__(self):
        return str(self)
    
    def to_dict(self)->Mapping[str, torch.Tensor|nn.Module]:
        return {key:value for key, value in self.items()}

    def clone(self)->'BufferDict':
        data = {key:value.clone() for key, value in self.items()} # type: ignore
        return BufferDict(data)