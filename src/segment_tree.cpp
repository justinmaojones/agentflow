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

    void update_tree2_c(int idx, double val, double* array, const int n, double* sumtree) {
        array += idx;
        idx = (idx + n)/2; // idx of parent in sumtree
        double diff = val - *array;
        *array = val;
        sumtree += idx;
        while(idx > 0) {
            *sumtree += diff;
            sumtree -= idx - (idx/2); // move to parent (idx rounds down)
            idx /= 2; // idx rounds down
        }
    }

    void update_tree_multi2_c(
            int* idxs, const int I,
            double* vals, const int V, 
            double* array, const int n,
            double* sumtree) {

        const double* vals0 = vals;
        int v = 0;
        for(int i=0; i<I; i++) {
            update_tree2_c(*idxs,*vals,array,n,sumtree);
            idxs++;
            vals++;
            v++;

            // if V < I, cycle through vals
            if(v==V){ 
                vals = (double*) vals0;
                v = 0;
            }
        }
    }

    void update_tree3_c(double* ref, double* val, double* array, double* sumtree, const int n) {
        int idx = (array - ref + n) / 2;
        double diff = *val - *array;
        *array = *val;
        sumtree += idx;
        while(idx > 0) {
            *sumtree += diff;
            sumtree -= idx - (idx/2);
            idx /= 2;
        }
    }

    void update_tree_multi3_c(double* ref, double* vals, double* array, const int m, double* sumtree, const int n) {
        for(int i=0; i<m; i++) {
            update_tree3_c(ref,vals,array,sumtree,n);
            vals++;
            array++;
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

    int get_prefix_sum_idx2_c(double val, double* array, double* sumtree, const int n) {
        int i = 1;
        double left_val;
        while(i < n) {
            int left = 2*i;
            if(left < n) {
                left_val = *(sumtree+left);
            } else {
                left_val = *(array+left-n);
            }
            if(val >= left_val) {
                i = left + 1; //right
                val -= left_val;
            } else {
                i = left;
            }
        }
        return i - n;
    }

    void get_prefix_sum_idx_multi2_c(int* outarray, double* vals, const int m, double* array, const int n, double* sumtree) {
        for(int i=0; i<m; i++) {
            *outarray = get_prefix_sum_idx2_c(*vals,array,sumtree,n);
            outarray++;
            vals++;
        }
    }
};



