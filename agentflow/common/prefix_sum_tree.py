import numpy as np

class SumTree(np.ndarray):
        
    def __new__(self,array):
        print('__new__',type(array))
        
        assert not isinstance(array,SumTree), "cannot build SumTree on another SumTree object"
        
        if array.base is None or array.size == array.base.size:
            return array.view(SumTree)
        
        else:
            # array base is a different size, so return a copy,
            # because it's important that base and self cover the same
            # memory space
            return array.copy().view(SumTree)  
        
    def __array_finalize__(self,array):
        print('__array_finalize__')
        
        if isinstance(self.base,SumTree):
            # inherit the same base and sum tree
            self._flat_base = self.base._flat_base
            self._sumtree = self.base._sumtree
        else:
            # initialize
            self._flat_base = array.view(np.ndarray).ravel()
            self._sumtree = np.zeros_like(self._flat_base)
            
        assert self.size == self.base.size, "self and base should be the same size"
            
        
    def __array_wrap__(self, out_arr, context=None):
        # any op that manipulates the array, other than setting values, 
        # should return an ndarray
        print('In __array_wrap__:')
        return super(SumTree, self).__array_wrap__(self, out_arr, context).view(np.ndarray)
    
    def __setitem__(self,idx,val):
        diffs = self.__array__()[idx]-val
        self.__array__()[idx] = val
        
    def __getitem__(self,idx):
        output = super(SumTree,self).__getitem__(idx)
        if output.size == self.size and self.base is output.base:
            return output
        else:
            return output.view(np.ndarray)
    
