import numpy as np
import agentflow.buffers.segment_tree_c as segment_tree

class SumPrefixTree(np.ndarray):
        
    def __new__(self,array):
        
        assert not isinstance(array,SumPrefixTree), "cannot build SumPrefixTree on another SumPrefixTree object"
        
        if (array.base is None or array.size == array.base.size) and array.flags['C_CONTIGUOUS']:
            return array.view(SumPrefixTree)
        
        else:
            # array base is a different size, so return a copy,
            # because it's important that base and self cover the same
            # memory space
            return array.copy().view(SumPrefixTree)  
        
    def __array_finalize__(self,array):
        
        if isinstance(self.base,SumPrefixTree):
            # inherit the same base and sum tree
            self._flat_base = self.base._flat_base
            self._indices = self.base._indices.reshape(array.shape)
            self._sumtree = self.base._sumtree
        else:
            # initialize
            self._flat_base = array.view(np.ndarray).ravel()
            self._indices = np.arange(array.size, dtype=np.int32).reshape(array.shape)
            self._sumtree = np.zeros_like(self._flat_base)
            
        assert self.size == self.base.size, "self and base should be the same size"
            
        
    def __array_wrap__(self, out_arr, context=None):
        # any op that manipulates the array, other than setting values, 
        # should return an ndarray
        return super(SumPrefixTree, self).__array_wrap__(out_arr, context).view(np.ndarray)
    
    def __setitem__(self,idx,val):
        #self.__array__()[idx] = val
        indices = np.ascontiguousarray(self._indices[idx]).ravel()
        values = np.ascontiguousarray(val,dtype=self._flat_base.dtype).ravel()
        segment_tree.update_tree_multi2(
                indices, values, self._flat_base, self._sumtree)

    def __getitem__(self,idx):
        output = super(SumPrefixTree,self).__getitem__(idx)
        if output.size == self.size and self.base is output.base:
            return output
        else:
            return output.view(np.ndarray)
    
    def get_prefix_sum_id(self,prefix_sum):
        prefix_sum_flat = np.ascontiguousarray(prefix_sum).ravel()
        output = np.zeros(prefix_sum.size,dtype=np.int32)
        segment_tree.get_prefix_sum_multi_idx2(output,prefix_sum_flat,self,self._sumtree)
        return output.reshape(prefix_sum.shape)
