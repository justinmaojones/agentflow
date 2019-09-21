




env = Env()
obs = env.reset()
state = preprocess(obs)

state_shape = state.shape[1:]
action_shape = []
dqda_clipping = 1
clip_norm = False

ddpg = DDPG(state_shape,action_shape,policy_fn,q_fn,dqda_clipping,clip_norm)

class Trainer(object):

    def __init__(self,env,agent,replay_buffer,preprocess_fn=None):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.preprocess_fn = preprocess_fn

    def preprocess(self,obs):
        if self.preprocess_fn is not None:
            return preprocess(obs)
        else:
            return obs 

    def reset(self):
        self.state = self.preprocess(self.env.reset())

    def step(self,sess):
        action = self.agent.act(sess,self.state)
        obs, reward, done, info = self.env.step(action)
        self.replay_buffer.append({'state':state,'action':action,'reward':reward,'done':done})
        self.state = self.preprocess(obs)

    def update(self,sess,batchsize=None,**kwargs):
        batchsize = batchsize or len(self.replay_buffer)-1
        training_sample = self.replay_buffer.sample(batchsize)
        self.agent.update(sess,training_sample,**kwargs)
