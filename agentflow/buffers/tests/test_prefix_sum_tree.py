import unittest
import numpy as np
from agentflow.buffers.prefix_sum_tree import PrefixSumTree

class TestPrefixSumTree(unittest.TestCase):

    def round_up_to_nearest_power_of_two(self,x):
        y = x
        k = 0
        while y > 0:
            y >>= 1
            k += 1
        return 2**k

    def test_round_up_to_nearest_power_of_two(self):
        self.assertEqual(self.round_up_to_nearest_power_of_two(1),2)
        self.assertEqual(self.round_up_to_nearest_power_of_two(2),4)
        self.assertEqual(self.round_up_to_nearest_power_of_two(3),4)
        self.assertEqual(self.round_up_to_nearest_power_of_two(4),8)

    def run_test_get_prefix_sum_id(self,x,sum_tree):
        # test get_prefix_sum_id
        n = len(sum_tree)
        n_up = self.round_up_to_nearest_power_of_two(n)
        #n_up = (n//2)*4 # round up to nearest power of 2
        j = (n_up - n) % n # offset due to array length not being power of 2
        xsum = x[j]
        for i in range(int(sum(x))):
            if i >= xsum:
                j = (j+1) % n
                xsum += x[j]
            self.assertEqual(j, sum_tree.get_prefix_sum_id(i)[0])


    def test_sum_tree(self):

        for size in range(1,16):
            x = np.random.choice(20,size=size)
            x += 1 # ensure positive
            sum_tree = PrefixSumTree(size)

            for i,v in enumerate(x):
                sum_tree[i] = v

            # test set
            for i in range(size):
                self.assertEqual(sum_tree[i], x[i])

            # test sum
            self.assertEqual(sum_tree.sum(), x.sum())

            # test get_prefix_sum_id
            self.run_test_get_prefix_sum_id(x,sum_tree)

    def test_sum_tree_init_array(self):

        for size in range(1,16):
            x = np.random.choice(20,size=size)
            x += 1 # ensure positive
            sum_tree = PrefixSumTree(x)

            # test set
            for i in range(size):
                self.assertEqual(sum_tree[i], x[i])

            # test sum
            self.assertEqual(sum_tree.sum(), x.sum())

            # test get_prefix_sum_id
            self.run_test_get_prefix_sum_id(x,sum_tree)

    def test_sum_tree_after_multiple_sets(self):

        for size in range(1,16):
            x = np.random.choice(50,size=size)
            x += 1 # ensure positive
            sum_tree = PrefixSumTree(size)

            for i,v in enumerate(x):
                sum_tree[i] = v

            x = np.random.choice(50,size=size)
            x += 1 # ensure positive
            for i,v in enumerate(x):
                sum_tree[i] = v

            # test set
            for i in range(size):
                self.assertEqual(sum_tree[i], x[i])

            # test sum
            self.assertEqual(sum_tree.sum(), x.sum())

            # test get_prefix_sum_id
            self.run_test_get_prefix_sum_id(x,sum_tree)


if __name__ == '__main__':
    unittest.main()
