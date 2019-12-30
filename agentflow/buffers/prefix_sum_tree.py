import numpy as np
import agentflow.buffers.prefix_sum_tree_methods as prefix_sum_tree_methods

class PrefixSumTree(np.ndarray):
        
    def __new__(self,shape_or_array):

        if isinstance(shape_or_array,PrefixSumTree):
            return shape_or_array
        
        elif isinstance(shape_or_array,np.ndarray):
            array = np.zeros(shape_or_array.shape,dtype=np.float64).view(PrefixSumTree)
            array.ravel()[:] = shape_or_array.ravel()
            return array
        else:
            return np.zeros(shape_or_array,dtype=np.float64).view(PrefixSumTree)
            
        
    def __array_finalize__(self,array):
        
        if isinstance(self.base,PrefixSumTree):
            # inherit the same base and sum tree
            self._flat_base = self.base._flat_base
            self._indices = self.base._indices.reshape(array.shape)
            self._sumtree = self.base._sumtree
        else:
            # initialize
            self._flat_base = array.view(np.ndarray).ravel()
            self._indices = np.arange(array.size, dtype=np.int32).reshape(array.shape)
            self._sumtree = np.zeros_like(self._flat_base)
            
        
    def __array_wrap__(self, out_arr, context=None):
        # any op that manipulates the array, other than setting values, 
        # should return an ndarray
        return super(PrefixSumTree, self).__array_wrap__(out_arr, context).view(np.ndarray)
    
    def __setitem__(self,idx,val):
        indices = np.ascontiguousarray(self._indices[idx]).ravel()
        values = np.ascontiguousarray(val,dtype=self._flat_base.dtype).ravel()
        prefix_sum_tree_methods.update_disjoint_tree_multi(
                indices, values, self._flat_base, self._sumtree)

    def __getitem__(self,idx):
        output = super(PrefixSumTree,self).__getitem__(idx)
        if output.size == self.size and self.base is output.base:
            # if the base is the same and we have the entire array
            # then it is safe to return as a PrefixSumTree object
            return output
        else:
            # otherwise, not safe, return a normal ndarray
            return output.view(np.ndarray)
    
    def get_prefix_sum_id(self,prefix_sum):
        # ensure prefix sum is the correct type
        prefix_sum = np.ascontiguousarray(prefix_sum,dtype=self.dtype)
        prefix_sum_flat = prefix_sum.ravel()
        # init return array
        output = np.zeros(prefix_sum.size,dtype=np.int32)
        # get ids
        prefix_sum_tree_methods.get_prefix_sum_multi_idx2(output,prefix_sum_flat,self._flat_base,self._sumtree)
        return output.reshape(prefix_sum.shape)

    def sample(self,nsamples=1):
        # sample priority values in the cumulative sum
        vals = (self.sum() * np.random.rand(nsamples)).astype(self.dtype)
        # init return array
        output = np.zeros(nsamples,dtype=np.int32)
        # get sampled ids
        prefix_sum_tree_methods.get_prefix_sum_multi_idx2(output,vals,self._flat_base,self._sumtree)
        return output


    def __sum__(self):
        if len(self) == 1:
            return self[0]
        else:
            return self._sumtree[1]

    def sum(self,*args,**kwargs):
        if len(args) == 0 and len(kwargs) == 0 and len(self) > 1:
            return self._sumtree[1]
        else:
            return super(PrefixSumTree, self).sum(*args,**kwargs)

