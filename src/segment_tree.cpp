#include <math.h>
#include <stdio.h>
#include "segment_tree.h"

namespace segment_tree {
    void update_tree_c(int idx, double val, double* array, const int n) {
        array += idx + n;
        idx += n;
        val -= *array;
        while(idx > 0) {
            *array += val; //update the value (by difference)
            array -= idx - (idx/2); //update the pointer
            idx /= 2;
        }
    }

    void update_tree_multi_c(int* idxs, double* vals, const int m, double* array, const int n) {
        // int* idxs: indices to update
        // double* vals: values for update
        // const int m: number of indices to update (i.e. length of idxs array)
        // double* array: sum_tree array to update
        // const int n: length of array leaves (i.e. since array is a sum_tree, it is array.size()/2)
        for(int i=0; i<m; i++) {
            update_tree_c(*idxs,*vals,array,n);
            idxs++;
            vals++;
        }
    }

    int get_prefix_sum_idx_c(double val, double* array, const int n) {
        int i = 1;
        while(i<n) {
            int left = 2*i;
            int right = 2*i+1;
            if(val >= *(array+left)) {
                i = right;
                val -= *(array+left);
            } else {
                i = left;
            }
        }
        return i - n;
    }

    void get_prefix_sum_idx_multi_c(int* outarray, double* vals, const int m, double* array, const int n) {
        // double* outarray: output array
        // double* vals: values for update
        // const int m: number of indices to update (i.e. length of idxs array)
        // double* array: sum_tree array to update
        // const int n: length of array leaves (i.e. since array is a sum_tree, it is array.size()/2)
        for(int i=0; i<m; i++) {
            *outarray = get_prefix_sum_idx_c(*vals,array,n);
            outarray++;
            vals++;
        }
    }
};
