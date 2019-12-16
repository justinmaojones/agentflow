# cimport the Cython declarations for numpy
cimport numpy as np
from cython.parallel import prange

# if you want to use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "segment_tree.h" namespace "segment_tree" nogil:
    void update_tree_c(int idx, double val, double* array, const int n);

cdef extern from "segment_tree.h" namespace "segment_tree" nogil:
    void update_tree_multi_c(int* idxs, double* val, const int m, double* array, const int n);

def update_tree(
            int idx,
            double val,
            np.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;

    update_tree_c(idx, val, &array[0], n);

def update_tree_multi(
            np.ndarray[int, ndim=1, mode="c"] idxs not None,
            np.ndarray[double, ndim=1, mode="c"] vals not None,
            np.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;
    cdef int m = idxs.shape[0];

    update_tree_multi_c(&idxs[0], &vals[0], m, &array[0], n);

