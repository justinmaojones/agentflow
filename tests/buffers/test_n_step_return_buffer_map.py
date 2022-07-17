import numpy as np
import unittest

from agentflow.buffers import NStepReturnBuffer
from agentflow.buffers import PrioritizedBufferMap 


class TestNStepReturnBuffer(unittest.TestCase):

    def test_all(self):
        B = 2
        T = 500
        NSTEPS = 10
        gamma = 0.99
        buf = PrioritizedBufferMap(1000,eps=1e-12,alpha=1.0)
        pub = NStepReturnBuffer(buf,n_steps=NSTEPS,gamma=gamma)

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
        buf_priority = buf._sumtree
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


if __name__ == '__main__':
    unittest.main()
