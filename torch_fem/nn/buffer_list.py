import torch 
import torch.nn as nn 
from typing import Optional, Iterable, List, Union

class BufferList(nn.Module):
    def __init__(self, data:Optional[Iterable[torch.Tensor]] = None):
        super().__init__()
        if data is None:
            data = {}
    
        for i, value in enumerate(data): 
            self.register_buffer(str(i), value)

        self._length = len(data) # type: ignore
        
    def as_parameter(self, key:int):
        """Convert a buffer to a parameter"""
        buffer = self._buffers.pop(str(key)) 
        self.register_parameter(str(key), buffer) # type: ignore
        
    def as_buffer(self, key:int):
        """Convert a parameter to a buffer"""
        parameter = self._parameters.pop(str(key))
        self.register_buffer(str(key), parameter)

    def append(self, value:torch.Tensor):
        self.register_buffer(str(len(self)), value)
        self._length += 1
    
    def insert(self, index:int, value:torch.Tensor):
        for i in range(len(self), index, -1):
            last_key = str(i - 1)
            if last_key in self._buffers:
                self.register_buffer(str(i), self._buffers[last_key])
            else: # in parameters
                self.register_parameter(str(i), self._parameters[last_key])
        key = str(index)
        if key in self._buffers:
            self.register_buffer(key, value)
        elif key in self._parameters:
            self.register_parameter(key, value) # type: ignore
        else:
            self.register_buffer(key, value) # append to the end

        self._length += 1

    def pop(self, index:int=-1)->torch.Tensor:
        if index < 0:
            index += len(self)
        assert index >=0 and index < len(self), f"Index {index} out of range"
        value = self._buffers.pop(str(index))
        for i in range(index+1, len(self)):
            last_key = str(i)
            if last_key in self._buffers:
                self.register_buffer(str(i-1), self._buffers[last_key])
            else:
                self.register_parameter(str(i-1), self._parameters[last_key])
        self._length -= 1
        return value # type: ignore
    
    def __hash__(self):
        return hash(tuple(self[i] for i in range(len(self))))

    def __getitem__(self, index:int|slice|torch.Tensor)->Union[torch.Tensor,'BufferList']: # type: ignore
        if isinstance(index, int):
            assert index >=0 and index < len(self), f"Index {index} out of range"
            key = str(index)
            if key in self._buffers:
                return self._buffers[key] # type:ignore
            else:
                return self._parameters[key] # type:ignore
        elif isinstance(index, slice):
            indices = index.indices(len(self))
            assert indices[0] >=0 and indices[1] <= len(self), f"Index {indices} out of range"
            result = []
            for i in range(*indices):
                key = str(i)
                if key in self._buffers:
                    result.append(self._buffers[key])
                else:
                    result.append(self._parameters[key])
            return BufferList(result)
        elif isinstance(index, torch.Tensor):
            assert (index >=0).all() and (index < len(self)).all(), f"Index {index} out of range"
            result = []
            for i in index:
                key = str(i.item())
                if key in self._buffers:
                    result.append(self._buffers[key])
                else:
                    result.append(self._parameters[key])
            return BufferList(result)
        else:
            raise TypeError(f"Index must be an integer, a slice or a tensor, not {type(index)}")
    
    def __setitem__(self, index:int, value:torch.Tensor):
        assert index >=0 and index < len(self), f"Index {index} out of range"
        self.register_buffer(str(index), value)
    
    def __delitem__(self, index:int):
        self.pop(index)
    
    def __len__(self)->int:
        return self._length
    
    def __iter__(self)->Iterable[torch.Tensor]:
        self._counter= 0
        return self 
    
    def __next__(self)->torch.Tensor:
        if self._counter >= len(self):
            raise StopIteration()
        
        key = str(self._counter)
        if key in self._buffers:
            value = self._buffers[key]
        else:
            value = self._parameters[key]

        self._counter += 1

        return value
    
    def __contains__(self, value:torch.Tensor)->bool:
        return value in self._buffers.values() or value in self._parameters.values()
    
    def __includes__(self, value:torch.Tensor)->bool:
        return value in self._buffers.values() or value in self._parameters.values()

    def item(self)->torch.Tensor:
        assert len(self) == 1, "BufferList must contain exactly one element"
        return self[0]
    
    def is_floating_point(self)->bool:
        return any(map(lambda x:x.is_floating_point(), iter(self)))

    def is_complex(self)->bool:
        return any(map(lambda x:x.is_complex(), iter(self)))

    @property
    def dtype(self):
        return next(iter(self)).dtype # type: ignore

    @property
    def device(self):
        return next(iter(self)).device # type: ignore
    
    def __str__(self):
        return f"""BufferList[
        {', '.join(map(str,iter(self)))} 
        ]""" 
      
    def __repr__(self):
        return str(self)
    
    def to_list(self)->List[torch.Tensor]:
        return list(iter(self))

    def clone(self)->'BufferList':
        return BufferList([value.clone() for value in self])
    

if __name__ == '__main__':
    x = torch.rand(3)
    y = torch.rand(1)
    b = BufferList([x, y])
    # print(b)
    # print(b.pop())
    # print(b)
    print(b)
    b.pop()
    print(b)
    b.insert(0, y)
    print(b._buffers)
    print(b)
    print(b.to_list())
    print(b.clone())