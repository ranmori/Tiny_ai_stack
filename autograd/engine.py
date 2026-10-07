# the autograd engine: a scalar Value that computes its own gradients
import math

class Value:
    # a scalar value, and a node in a computational graph
    #     """A scalar value that remembers the operation that created it,
    # so it can compute gradients automatically (reverse-mode autodiff)
    def __init__(self, data, _children=(), _op='', label=''):
        
        self.data = data # the scalar value
        self.grad = 0.0 # the gradient of this value
        self._backward = lambda: None # a function to compute the gradient of this value with respect to its inputs
        self._prev = set(_children) # the set of input nodes to this value
        self._op = _op    # the operation that produced this value
        self.label = label  # a label for debugging purposes

    def __add__(self, other):
        # add two values
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # compute the gradient of the output with respect to the inputs
            # # Chain rule: if out = self + other, then
            # d(out)/d(self) = 1  and  d(out)/d(other) = 1
            # out.grad is "how much does the final loss change per unit change in out"
            # (already computed by whoever came after `out` in the graph)
            # so self's share of that sensitivity is 1 * out.grad, same for other
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        # multiply two values
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # compute the gradient of the output with respect to the inputs
            # Chain rule: if out = self * other, then
            # d(out)/d(self) = other  and  d(out)/d(other) = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # raise a value to a power
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # compute the gradient of the output with respect to the input
            # Chain rule: if out = self ** other, then
            # d(out)/d(self) = other * self ** (other - 1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out  



    # activation functions

    def tanh(self):
        # tanh is a smooth, differentiable function that squashes input values to the range (-1, 1)
        # compute the tanh of the value
        t = math.tanh(self.data)
        out = Value(t, (self,), 'Tanh')

        def _backward():
            # compute the gradient of the output with respect to the input
            # Chain rule: if out = Tanh(self), then
            # d(out)/d(self) = 1 - out^2
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out
    def relu(self):  #recitified linear unit
        # keeps positive numbers and sets negative numbers to zero
        # compute the ReLU of the value
        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            # compute the gradient of the output with respect to the input
            # Chain rule: if out = ReLU(self), then
            # d(out)/d(self) = 1 if self > 0 else 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        # sigmoid is a smooth, differentiable function that squashes input values to the range (0, 1)
        # compute the sigmoid of the value
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'Sigmoid')

        def _backward():
            # compute the gradient of the output with respect to the input
            # Chain rule: if out = Sigmoid(self), then
            # d(out)/d(self) = out * (1 - out)
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward

        return out

    
    def backward(self):
        # topological order all of the children in the graph
        # so that we can go one variable at a time and apply the chain rule to get its gradient
        topo = []
        # the visited set is used to avoid visiting the same node multiple times
        visited = set()
        # define a recursive function to build the topological order
        def build_topo(v):
            # if the node has not been visited yet, add it to the visited set and recursively visit its children
            if v not in visited:
                # add the node to the visited set
                visited.add(v)
                # recursively visit the children of the node
                for child in v._prev:
                    build_topo(child)
                    # after visiting all the children, add the node to the topological order
                topo.append(v)
                # return the topological order
        build_topo(self)
        
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()



    # convenience methods for printing the value and gradient
    def __neg__(self):
        return self * -1   # -a  ==  a * -1
    def __sub__(self, other):
        return self + (-other) # a - b  ==  a + (-b)
    def __radd__(self, other):
        return self + other # other + self  ==  self + other  # handles `5 + a` (plain number first)

    def __rmul__(self, other):
        return self * other # other * self  ==  self * other  # handles `5 * a`
    def __rsub__(self, other):
        return other + (-self) # other - self  ==  other + (-self)  # handles `5 - a`
    def __truediv__(self, other):
        return self * other**-1 # a / b  ==  a * b**-1
    
    def __rtruediv__(self, other):
        # divide a value by another value
        return self**-1 * other  # a / b  ==  a * b**-1
    

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
