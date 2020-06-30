import numpy as np
from agentflow.buffers.buffer_map import BufferMap
from agentflow.buffers.prioritized_buffer_map import PrioritizedBufferMap 

class DelayedBufferMapPublisher(BufferMap):

    def __init__(self,max_length=2**20,publish_indicator_key="done",add_return_loss=False):
        super(DelayedBufferMapPublisher,self).__init__(max_length)
        self._publish_indicator_key = publish_indicator_key
        self._count_since_last_publish = None 
        self._published = []
        self.add_return_loss = add_return_loss

    def compute_returns(self,data,gamma=0.99):
        rewards = data['reward']
        T = rewards.shape[-1]
        returns = []
        R = 0
        for t in reversed(range(T)):
            r = rewards[...,t:t+1]
            R = r + gamma*R
            returns.append(R)
        return np.concatenate(returns[::-1],axis=-1)

    def publish(self,data):
        if self._count_since_last_publish is None:
            self._count_since_last_publish = np.zeros_like(data[self._publish_indicator_key]).astype(int)
        self._count_since_last_publish += 1
        assert self._count_since_last_publish.max() <= self.max_length, "delay cannot exceed size of buffer"
    
        should_publish = data[self._publish_indicator_key]
        assert should_publish.ndim == 1, "expected data['%s'] to have ndim==1, but found ndim==%d" % (self._publish_indicator_key, should_publish.ndim)

        output = []
        if np.sum(should_publish) > 0:
            # at least one record should be published
            idx = np.arange(len(should_publish))[should_publish==1]
            for i in idx:
                # cannot retrieve sequence larger than buffer size
                seq_size = min(self._count_since_last_publish[i],self._n)
                published = self.tail(seq_size,batch_idx=[i])
                if self.add_return_loss:
                    published['returns'] = self.compute_returns(published)
                output.append(published)
                self._count_since_last_publish[i] = 0
        return output

    def append(self,data):
        super(DelayedBufferMapPublisher,self).append(data)
        return self.publish(data)

class DelayedBufferMap(BufferMap):
    
    def __init__(self,max_length=2**20,delayed_buffer_max_length=None,publish_indicator_key="done",add_return_loss=False):
        super(DelayedBufferMap,self).__init__(max_length)
        self._publish_indicator_key = publish_indicator_key
        self._delayed_buffer_max_length = delayed_buffer_max_length if delayed_buffer_max_length is not None else max_length
        self._delayed_buffer_map = DelayedBufferMapPublisher(self._delayed_buffer_max_length,publish_indicator_key,add_return_loss)

    def append(self,data):
        published = self._delayed_buffer_map.append(data)
        for seq in published:
            super(DelayedBufferMap,self).append_sequence(seq)

    def append_sequence(self,data):
        raise NotImplementedError

class DelayedPrioritizedBufferMap(PrioritizedBufferMap):
    
    def __init__(self,delayed_buffer_max_length=None,publish_indicator_key="done",add_return_loss=False,**kwargs):
        super(DelayedPrioritizedBufferMap,self).__init__(**kwargs)
        self._publish_indicator_key = publish_indicator_key
        self._delayed_buffer_max_length = delayed_buffer_max_length if delayed_buffer_max_length is not None else self.max_length
        self._delayed_buffer_map = DelayedBufferMapPublisher(self._delayed_buffer_max_length,publish_indicator_key,add_return_loss)

    def append(self,data):
        published = self._delayed_buffer_map.append(data)
        for seq in published:
            super(DelayedPrioritizedBufferMap,self).append_sequence(seq)

    def append_sequence(self,data):
        raise NotImplementedError

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def test_all(self):
            n = 10
            buf = DelayedBufferMap(n)

            # append data, but nothing ready to publish yet
            x = {
                'x': np.array([1,2]),
                'done': np.array([0,0]),
            }
            buf.append(x)
            self.assertEqual(len(buf.buffers), 0)
            self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([1,1])))

            # append data, element 1 ready to publish 
            x = {
                'x': np.array([3,4]),
                'done': np.array([1,0]),
            }
            buf.append(x)
            self.assertTrue(np.all(buf.buffers['x'].buffer[...,:2]==np.array([[1,3]])))
            self.assertTrue(np.all(buf.buffers['done'].buffer[...,:2]==np.array([[0,1]])))
            self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([0,2])))

            # append data, both elements ready to publish 
            x = {
                'x': np.array([5,6]),
                'done': np.array([1,1]),
            }
            buf.append(x)
            self.assertTrue(np.all(buf.buffers['x'].buffer==np.array([[1,3,5,2,4,6,0,0,0,0]])))
            self.assertTrue(np.all(buf.buffers['done'].buffer==np.array([[0,1,1,0,0,1,0,0,0,0]])))
            self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([0,0])))

            # append data, wrap around 
            for i in range(0,9):
                buf.append({'x': np.array([i,i+1]), 'done': np.array([0,0])})

            self.assertTrue(np.all(buf.buffers['x'].buffer==np.array([[1,3,5,2,4,6,0,0,0,0]])))
            self.assertTrue(np.all(buf.buffers['done'].buffer==np.array([[0,1,1,0,0,1,0,0,0,0]])))
            self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([9,9])))

            x = {
                'x': np.array([9,10]),
                'done': np.array([1,0]),
            }
            buf.append(x)
            self.assertTrue(np.all(buf.buffers['x'].buffer==np.array([[4, 5, 6, 7, 8, 9, 0, 1, 2, 3]])))
            self.assertTrue(np.all(buf._delayed_buffer_map._count_since_last_publish==np.array([0,10])))



    unittest.main()

