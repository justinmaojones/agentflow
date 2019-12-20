# cimport the Cython declarations for numpy
cimport numpy as cnp
import numpy as np
from cython.parallel import prange

# if you want to use the Numpy-C-API from Cython
cnp.import_array()

# cdefine the signature of our c function
cdef extern from "segment_tree.h" namespace "segment_tree" nogil:
    void update_tree_c(int idx, double val, double* array, const int n);

cdef extern from "segment_tree.h" namespace "segment_tree" nogil:
    void update_tree_multi_c(int* idxs, double* val, const int m, double* array, const int n);

cdef extern from "segment_tree.h" namespace "segment_tree" nogil:
    int get_prefix_sum_idx_c(double val, double* array, const int n);

cdef extern from "segment_tree.h" namespace "segment_tree" nogil:
    void get_prefix_sum_idx_multi_c(int* outarray, double* vals, const int m, double* array, const int n); 

def update_tree(
            int idx,
            double val,
            cnp.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;

    update_tree_c(idx, val, &array[0], n);

def update_tree_multi(
            cnp.ndarray[int, ndim=1, mode="c"] idxs not None,
            cnp.ndarray[double, ndim=1, mode="c"] vals not None,
            cnp.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;
    cdef int m = idxs.shape[0];

    update_tree_multi_c(&idxs[0], &vals[0], m, &array[0], n);

def get_prefix_sum_idx(
            double val,
            cnp.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;

    return get_prefix_sum_idx_c(val, &array[0], n);

def get_prefix_sum_multi_idx(
            cnp.ndarray[int, ndim=1, mode="c"] output not None,
            cnp.ndarray[double, ndim=1, mode="c"] vals not None,
            cnp.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;
    cdef int m = vals.shape[0];

    get_prefix_sum_idx_multi_c(&output[0], &vals[0], m, &array[0], n);

    return output

def get_prefix_sum_multi_idx_parallel(
            cnp.ndarray[int, ndim=1, mode="c"] output not None,
            cnp.ndarray[double, ndim=1, mode="c"] vals not None,
            cnp.ndarray[double, ndim=1, mode="c"] array not None):

    cdef int n = array.shape[0]//2;
    cdef int m = vals.shape[0];
    cdef int i;

    for i in prange(m,nogil=True):
        output[i] = get_prefix_sum_idx_c(vals[i], &array[0], n);

    return output
