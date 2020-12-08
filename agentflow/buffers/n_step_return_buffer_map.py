import numpy as np
from agentflow.buffers.buffer_map import BufferMap
from collections import deque

class NStepReturnPublisher(object):
    
    def __init__(self,
            buffer_map,
            n_steps=1,
            gamma=0.99,
            reward_key="reward",
            done_indicator_key="done",
            delayed_keys=['state2'],
            ):

        if not isinstance(buffer_map, BufferMap):
            raise TypeError("buffer_map must be a type of BufferMap")

        self._buffer_map = buffer_map
        self._n_steps = n_steps
        self._gamma = gamma
        self._reward_key = reward_key
        self._done_indicator_key = done_indicator_key
        self._delayed_keys = delayed_keys
        self._delayed_buffer = BufferMap(n_steps)
        self._kwargs_queue = deque()
        self._returns = None
        self._dones = None
        self._discounts = gamma**np.arange(n_steps)[:,None]

    def compute_returns_and_dones(self):
        rewards = self._delayed_buffer._buffers[self._reward_key].tail(self._n_steps)
        dones = self._delayed_buffer._buffers[self._done_indicator_key].tail(self._n_steps)
        dones_shift_right_one = np.roll(dones,1,axis=0)
        dones_shift_right_one[0] = 0
        mask = (1-np.maximum.accumulate(dones_shift_right_one,axis=0))
        self.mask = mask
        returns = np.sum(rewards*self._discounts*mask,axis=0)
        return_dones = dones.max(axis=0)
        return returns, return_dones

    def append(self,data,**kwargs):
        self._delayed_buffer.append(data)
        self._kwargs_queue.append(kwargs)
        if len(self._delayed_buffer) == self._n_steps:
            returns, dones = self.compute_returns_and_dones()
            data_to_publish = {
                self._reward_key: returns,
                self._done_indicator_key: dones
            }
            for k in data:
                if k not in data_to_publish and k not in self._delayed_keys:
                    data_to_publish[k] = self._delayed_buffer._buffers[k].get(0)
            for k in self._delayed_keys:
                data_to_publish[k] = data[k] 
            delayed_kwargs = self._kwargs_queue.popleft()
            self._buffer_map.append(data_to_publish,**delayed_kwargs)

        elif len(self._delayed_buffer) >= self._n_steps:
            raise Exception("should never get here") 
        else:
            pass # nothing to append

    def __len__(self):
        return len(self._buffer_map)

    def sample(self,nsamples,**kwargs):
        return self._buffer_map.sample(nsamples,**kwargs)

    def update_priorities(self,priority):
        return self._buffer_map.update_priorities(priority)

    def priorities(self):
        return self._buffer_map.priorities()


if __name__ == '__main__':
    import unittest
    from agentflow.buffers import PrioritizedBufferMap 

    class Test(unittest.TestCase):

        def test_all(self):
            B = 2
            T = 500
            NSTEPS = 10
            gamma = 0.99
            buf = PrioritizedBufferMap(1000,eps=0.0,alpha=1.0)
            pub = NStepReturnPublisher(buf,n_steps=NSTEPS,gamma=gamma)

            data = {
                'state': np.random.randn(T,B,2,3),
                'state2': np.random.randn(T,B,2,3),
                'reward': np.random.randn(T,B),
                'action': np.random.randn(T,B),
                'done': np.random.choice(2,size=(T,B)),
            }

            kwargs = []
            for t in range(T):
                if t % 2 == 0:
                    kwargs.append({'priority':1+np.random.randn(B)**2})
                else:
                    kwargs.append({})

            for t in range(T):
                x = {k: data[k][t] for k in data}
                pub.append(x,**kwargs[t])
                if t + 1 < NSTEPS:
                    self.assertEqual(len(buf),0)
                else:
                    self.assertEqual(len(buf),(t+1)-NSTEPS+1)

            done = data['done']
            reward = data['reward']
            returns = buf._buffers['reward']._buffer[:len(buf)]
            returns_dones = buf._buffers['done']._buffer[:len(buf)]

            state = data['state']
            state2 = data['state2']
            action = data['action']
            buf_state = buf._buffers['state']._buffer[:len(buf)]
            buf_state2 = buf._buffers['state2']._buffer[:len(buf)]
            buf_action = buf._buffers['action']._buffer[:len(buf)]
            buf_priority = buf._sum_tree
            for t in range(len(buf)):
                R = np.zeros(B)
                D = np.zeros(B)
                for s in range(NSTEPS)[::-1]:
                    R = reward[t+s] + gamma*(1-done[t+s])*R
                    D = np.maximum(D,done[t+s])
                self.assertAlmostEqual(
                        np.abs(R-returns[t]).max(),
                        0,
                        places=4)
                self.assertEqual(
                        np.abs(D-returns_dones[t]).max(),
                        0)
                self.assertAlmostEqual(
                        np.abs(action[t]-buf_action[t]).max(),
                        0,
                        places=4)
                self.assertAlmostEqual(
                        np.abs(state[t]-buf_state[t]).max(),
                        0,
                        places=4)
                self.assertAlmostEqual(
                        np.abs(state2[t+NSTEPS-1]-buf_state2[t]).max(),
                        0,
                        places=4)
                priority = np.ones(B) if len(kwargs[t]) == 0 else kwargs[t]['priority']
                self.assertAlmostEqual(
                        np.abs(priority-buf_priority.get(t)).max(),
                        0,
                        places=4)


    unittest.main()

