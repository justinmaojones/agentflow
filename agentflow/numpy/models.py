import numpy as np
from sklearn.linear_model import Ridge, LinearRegression


class Logs(object):

    def __init__(self):
        self.logs = {}

    def append(self,key,val):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(val)

    def append_dict(self,inp):
        for k in inp:
            self.append(k,inp[k])

    def stack(self,key=None):
        if key is None:
            return {key:self.stack(key) for key in self.logs}
        else:
            return np.stack(self.logs[key])

class Model(object):

    def learn(self,X,X2,r,done,w0,T=100,lr=1,gamma=0.99,alpha=1,**update_kwargs):
        w = w0.copy()
        logs = Logs()
        for t in range(T):
            y, err, w = self.update(X,X2,r,done,w,lr,gamma,alpha,**update_kwargs)
            loss = 0.5*(err**2).mean() + 0.5*alpha*np.dot(w,w)
            logs.append('y',y)
            logs.append('err',err)
            logs.append('loss',loss)
            logs.append('w',w)
        return logs

    def learn_sgd(self,X,X2,r,done,w0,T=100,lr=1,gamma=0.99,alpha=1,batchsize=10,**update_kwargs):
        w = w0.copy()
        logs = Logs()
        idx = np.arange(len(X))
        for t in range(T):
            np.random.shuffle(idx)
            i = idx[:batchsize]
            y, err, w = self.update(X[i],X2[i],r[i],done[i],w,lr,gamma,alpha,**update_kwargs)
            loss = 0.5*(err**2).mean() + 0.5*alpha*np.dot(w,w)
            logs.append('y',y)
            logs.append('err',err)
            logs.append('loss',loss)
            logs.append('w',w)
        return logs

class TimeDelayedModel(Model):

    def update_w_ema(self,w,ema=0.99):
        if not hasattr(self,'_w_ema'):
            self._w_ema = w.copy()
        else:
            self._w_ema = ema*self._w_ema + (1-ema)*w
        return self._w_ema

class TimeDelayedGradientDescent(TimeDelayedModel):

    def update(self,X,X2,r,done,w,lr=1,gamma=0.99,alpha=1,ema=0.99,**kwargs):
        self.update_w_ema(w,ema)
        y = np.dot(X,w)
        y2 = np.dot(X2,self._w_ema)
        err = y - (r+(1-done)*gamma*y2)
        g = np.dot(err,X)/len(X) + alpha*w
        return y, err, w - lr*g

class TimeDelayedNewton(TimeDelayedModel):

    def update(self,X,X2,r,done,w,lr=1,gamma=0.99,alpha=1,ema=0.99,**kwargs):
        self.update_w_ema(w,ema)
        y = np.dot(X,w)
        y2 = np.dot(X2,self._w_ema)
        err = y - (r+(1-done)*gamma*y2)
        g = np.dot(err,X)/len(X) + alpha*w
        H = np.dot(X.T,X) + alpha*np.eye(len(w))
        Hinv = np.linalg.inv(H)
        v = np.dot(g,Hinv)
        return y, err, w - lr*v

class GradientDescent(Model):

    def update(self,X,X2,r,done,w,lr=1,gamma=0.99,alpha=1,**kwargs):
        y = np.dot(X,w)
        y2 = np.dot(X2,w)
        X_diff = X-gamma*(1-done[:,None])*X2
        err = y - (r+(1-done)*gamma*y2)
        g = np.dot(err,X_diff)/len(X) + alpha*w
        return y, err, w - lr*g

class Newton(Model):

    def update(self,X,X2,r,done,w,lr=1,gamma=0.99,alpha=1,**kwargs):
        y = np.dot(X,w)
        y2 = np.dot(X2,w)
        X_diff = X-gamma*(1-done[:,None])*X2
        err = y - (r+(1-done)*gamma*y2)
        g = np.dot(err,X_diff)/len(X) + alpha*w
        H = np.dot(X_diff.T,X_diff) + alpha*np.eye(len(w))
        Hinv = np.linalg.inv(H)
        v = np.dot(g,Hinv)
        return y, err, w - lr*v

class TimeDelayedGradientRidge(TimeDelayedModel):

    def update(self,X,X2,r,done,w,lr=1,gamma=0.99,alpha=1,ema=0.99,**kwargs):
        self.update_w_ema(w,ema)
        y = np.dot(X,w)
        y2 = np.dot(X2,self._w_ema)
        err = y - (r+(1-done)*gamma*y2)
        ridge = Ridge(alpha=alpha,fit_intercept=False)
        dydw = X
        ridge.fit(dydw,err)
        v = ridge.coef_
        return y, err, w - lr*v

class TimeDelayedStableUpdate(TimeDelayedModel):

    def update(self,X,X2,r,done,w,lr=1,gamma=0.99,alpha=1,ema=0.99,beta=1,**kwargs):
        self.update_w_ema(w,ema)
        y = np.dot(X,w)
        y2 = np.dot(X2,self._w_ema)
        err = y - (r+(1-done)*gamma*y2)
        g = np.dot(err,X)/len(X) 
        H = np.dot(X.T,X) + alpha*np.eye(len(w)) + beta*np.dot(X2.T,X2)
        Hinv = np.linalg.inv(H)

        """
        Can also be implemented this way

        X = np.concatenate([X,X2*beta**0.5],axis=0)
        err = np.concatenate([err,np.zeros(len(err))],axis=0)

        g = np.dot(err,X)
        H = np.dot(X.T,X) + alpha*np.eye(len(w))
        """

        v = np.dot(g,Hinv)
        return y, err, w - lr*v

