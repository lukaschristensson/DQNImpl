import numpy as np
import copy

activation_functions = {
    'ReLU': lambda x: x*(x>0),
    'sigmoid': lambda x: 1/(1+np.exp(-x)),
    None: lambda x: x
}
der_activation_functions = {
    'ReLU': lambda x: x > 0,
    'sigmoid': lambda x: activation_functions['sigmoid'](x) * (1 - activation_functions['sigmoid'](x)),
    None: lambda x: x/x
}
loss_functions = {
    'MSE': lambda y, y_h: .5 * np.mean((y-y_h)**2), # divided by two to make the der nicer
    'log_loss': lambda y, y_h: -sum(y*np.log(y_h) + (1-y)*np.log(1-y_h)) / len(y)
}
der_loss_functions = {
    'MSE': lambda y, y_h: y_h-y,
    'log_loss': lambda y, y_h: -((y/y_h) - (1-y)/(1-y_h))
}

class AdamOptimizer:
    def __init__(self, layers, b1=0.9, b2=0.999,  a=0.001, eps=1e-8):
        self.b1 = b1
        self.b2 = b2
        self.a = a
        self.eps = eps
        self.m = []
        self.v = []
        for l in layers:
            self.m.append(np.zeros_like(l))
            self.v.append(np.zeros_like(l))
        self.t = 0
    def opt(self, gts):
        self.t = self.t + 1
        ret_gts = []
        for i in range(len(gts)):
            self.m[i] = self.b1 * self.m[i] + (1-self.b1) * gts[i]
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * gts[i]**2
            m_h = self.m[i] / (1 - self.b1 ** self.t)
            v_h = self.v[i] / (1 - self.b2 ** self.t)
            gts[i][:] = self.a*m_h/(np.sqrt(v_h) + self.eps)
        return gts

class FCNN:
    def __init__(self, ls, act_fun='ReLU', output_act_fun=None, opt='Adam', loss_fun='MSE', learning_rate=0.001):
        self.act_fun = act_fun
        self.loss_fun = loss_fun
        self.output_act_fun = output_act_fun
        self.learning_rate = learning_rate
        self.ls = ls
        self.layers = []
        self.opt = opt
        for i in range(len(ls)-1):
            self.layers.append(np.random.random((ls[i]+1, ls[i+1])) * 2 - 1) # +1 for bias
        if opt != 'Adam' and opt is not None:
            raise Exception(f'Optimizer {opt} not implemented, please use Adam')
        self.optimizer = AdamOptimizer(self.layers, a=self.learning_rate)

    def __copy__(self):
        copy_nn = FCNN(self.ls, self.act_fun, self.output_act_fun, self.opt, self.loss_fun, self.learning_rate)
        copy_nn.layers = copy.deepcopy(self.layers)
        return copy_nn

    def forward(self, x, y=None):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[1] != self.ls[0]:
            raise Exception(f'Wrong X dimension: x_dim = {x.shape[1]}, input_dim = {self.ls[0]}')
        if y is not None and y.shape[1] != self.ls[-1]:
            raise Exception(f'Wrong Y dimension: y_dim = {len(y)}, output_dim = {self.ls[-1]}')

        act = activation_functions[self.act_fun]
        acts = [x]
        zs = []
        for l in self.layers[:-1]:
            x = np.c_[np.ones((x.shape[0], 1)), x] @ l
            zs.append(x)
            x = act(x)
            acts.append(x)
        y_pred = np.c_[np.ones((x.shape[0], 1)), x] @ self.layers[-1]
        zs.append(y_pred)
        y_pred = activation_functions[self.output_act_fun](y_pred)
        acts.append(y_pred)
        loss = None
        if y is not None:
            loss = loss_functions[self.loss_fun](y, y_pred)
            der_loss = der_loss_functions[self.loss_fun](y, y_pred)
            der_act = der_activation_functions[self.act_fun]
            der_output_act = der_activation_functions[self.output_act_fun]
            deltas = [der_loss * der_output_act(zs[-1])]
            grad = []

            for L in range(len(self.layers)):
                # logic to make the concatenation work smoothly when calculating the gradient
                bias_grad = np.mean(deltas[-1], axis=0)
                if bias_grad.ndim == 1:
                    bias_grad = bias_grad.reshape(1, -1)
                weight_grad = (deltas[-1].T @ acts[-2-L])
                if weight_grad.ndim == 1:
                    weight_grad = weight_grad.reshape(1, -1) / len(y)

                # calc grad
                grad.append(np.r_[bias_grad, weight_grad.T])
                # calc delta
                if L != len(self.layers)-1:
                    deltas.append(((deltas[-1] @ self.layers[-1-L][1:, :].T) * der_act(zs[-2-L])))

            # update layers
            if self.opt is not None:
                grad = self.optimizer.opt(grad[::-1])
            else:
                grad = grad[::-1]
                for g in grad:
                    g *= self.learning_rate
            for i in range(len(self.layers)):
                self.layers[i] -= grad[i]
        if loss is not None:
            return loss, y_pred
        return y_pred
