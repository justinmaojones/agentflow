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
        for(int i=0; i<m; i++) {
            update_tree_c(*idxs,*vals,array,n);
            idxs++;
            vals++;
        }
    }

};
