#ifndef SEGMENT_TREE_H
#define SEGMENT_TREE_H

namespace segment_tree {
    void update_tree_c(int idx, double val, double* array, const int n);
    void update_tree_multi_c(int* idxs, double* vals, const int m, double* array, const int n);
}

#endif
