

class Trainer:

    def __init__(self, env, agent, replay_buffer, preprocess_fn=None):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self._t

    def train_step(self):


    def step(self):
        action = self.agent.act(sess, self.state)
        obs, reward, done, info = self.env.step(action)
        self.replay_buffer.append({'state':state, 'action':action, 'reward':reward, 'done':done})
        self.state = self.preprocess(obs)

    def update(self, sess, batchsize=None, **kwargs):
        batchsize = batchsize or len(self.replay_buffer)-1
        training_sample = self.replay_buffer.sample(batchsize)
        self.agent.update(sess, training_sample, **kwargs)
