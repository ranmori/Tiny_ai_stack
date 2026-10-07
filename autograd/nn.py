#  a tiny  neural network library with autograd
import random
from engine import Value


# stack values into neural network layers
class Neuron:
    # a single neuron in a neural network
    def __init__(self, nin, nonlin=True):
        # initialize the weights and bias of the neuron
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # weights
        self.b = Value(0.0)  # bias
        self.nonlin = nonlin

    def __call__(self, x):
        # compute the output of the neuron given an input x
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # weighted sum + bias
        if self.nonlin:
            out = act.relu()  # apply activation function (ReLU)
        else:
            out = act
        return out

    def parameters(self):
        # return the parameters of the neuron (weights and bias)
        return self.w + [self.b]


class Layer:
    # a layer of neurons in a neural network
    def __init__(self, nin, nout, nonlin=True):
        # initialize the neurons in the layer
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        # compute the output of the layer given an input x
        out = [n(x) for n in self.neurons]  # apply each neuron to the input
        return out[0] if len(out) == 1 else out  # a single output comes back as a Value, not a list

    def parameters(self):
        # return the parameters of the layer (weights and biases of all neurons)
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    # a multi-layer perceptron (MLP) neural network
    def __init__(self, nin, nouts):
        # initialize the layers of the MLP
        sz = [nin] + nouts  # sizes of each layer
        # every layer uses ReLU except the last one, so the output can be any number (negative too)
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]  # create layers

    def __call__(self, x):
        # compute the output of the MLP given an input x
        for layer in self.layers:
            x = layer(x)  # apply each layer to the input
        return x

    def parameters(self):
        # return the parameters of the MLP (weights and biases of all layers)
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # reset the gradients of all parameters to zero
        for p in self.parameters():
            p.grad = 0.0

    