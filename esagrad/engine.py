import math

class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self.grad = 0.0
    self._backward = lambda: None

  def __repr__(self):
    return f"{self.data}"

  def __add__(self, other):
    out = Value(self.data + other.data, {self, other}, '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward

    return out

  def __sub__(self, other):
    return Value(self.data - other.data, {self, other}, '-')

  def __mul__(self, other):
    out = Value(self.data * other.data, {self, other}, '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ),'tanh')

    def _backward():
      self.grad += (1 - t**2)*out.grad

    out._backward = _backward

    return out

  def sigmoid(self):
    x = self.data
    s = 1/(math.exp(-x) + 1)
    out = Value(s, (self, ), 'sigmoid')

    def _backward():
      self.grad += s*(1-s)

    out._backward = _backward

    return out

  def relu(self):
    r = 0 if self.data <= 0 else self.data
    out = Value(r, (self,), 'relu')

    def _backward():
      self.grad += 0 if self.data <= 0 else 1

    out._backward = _backward

    return out
