
# Check the Value class against PyTorch: same expression, same data, same gradients
import pytest
import torch
from engine import Value


def check(expression, *inputs):
    # build the same expression twice, once with Value and once with torch tensors,
    # run backward on both, and compare the output and the gradient of every input
    vals = [Value(x) for x in inputs]
    out = expression(*vals)
    out.backward()

    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    expected = expression(*tensors)
    expected.backward()

    # forward pass
    assert out.data == pytest.approx(expected.item())
    # backward pass
    for v, t in zip(vals, tensors):
        assert v.grad == pytest.approx(t.grad.item())


# one operation at a time

def test_add():
    check(lambda a, b: a + b, 2.0, -3.0)

def test_mul():
    check(lambda a, b: a * b, 2.0, -3.0)

def test_pow():
    check(lambda a: a ** 3, 2.0)

def test_neg():
    check(lambda a: -a, 2.0)

def test_sub():
    check(lambda a, b: a - b, 2.0, -3.0)

def test_truediv():
    check(lambda a, b: a / b, 2.0, -3.0)


# a plain number mixed with a Value, on either side

def test_number_on_the_right():
    check(lambda a: a + 5, 2.0)
    check(lambda a: a * 5, 2.0)
    check(lambda a: a - 5, 2.0)
    check(lambda a: a / 5, 2.0)

def test_number_on_the_left():
    check(lambda a: 5 + a, 2.0)
    check(lambda a: 5 * a, 2.0)
    check(lambda a: 5 - a, 2.0)


# activation functions, on a negative and a positive input

@pytest.mark.parametrize('x', [-2.0, -0.5, 0.5, 2.0])
def test_tanh(x):
    check(lambda a: a.tanh(), x)

@pytest.mark.parametrize('x', [-2.0, -0.5, 0.5, 2.0])
def test_sigmoid(x):
    check(lambda a: a.sigmoid(), x)

@pytest.mark.parametrize('x', [-2.0, -0.5, 0.5, 2.0])
def test_relu(x):
    check(lambda a: a.relu(), x)


# whole graphs

def test_value_used_twice():
    # a feeds two operations, so its gradient must be accumulated with +=
    check(lambda a: a + a, 3.0)
    check(lambda a, b: a * b + a, 2.0, 3.0)

def test_neuron():
    # out = tanh(x1*w1 + x2*w2 + b)
    check(lambda x1, x2, w1, w2, b: (x1 * w1 + x2 * w2 + b).tanh(),
          2.0, 0.0, -3.0, 1.0, 6.88)

def test_bigger_expression():
    def expression(a, b):
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + f.sigmoid().tanh()
        return g
    check(expression, -4.0, 2.0)
