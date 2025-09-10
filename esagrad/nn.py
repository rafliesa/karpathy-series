import random

class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x, activation='tanh'):
     result = sum(wi*xi for wi,xi in zip(self.w, x)) + self.b
     return result.tanh()

  def parameters(self):
    return self.w + [self.b]


class Layer:

  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _n in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

  def __init__(self, nin, nouts, lr=0.01, epoch=20):
    self.lr = lr
    self.epoch = epoch
    sz = [nin] + nouts
    self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

  def __forward__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def fit(self, X, y):
    for i in range(self.epoch):
      ypred = [self.__forward__(x) for x in X]
      loss = sum([(ygt-yout)**2 for ygt, yout in zip(y, ypred)])

      loss.backward()

      for p in self.parameters():
        p.data -= self.lr * p.grad
        p.grad = 0

      print(i, loss.data)

  def predict(self,X):
    if len(X) == 1 :
      return self.__forward__(X[0])

    return [self.__forward__(x) for x in X]