# Autograd notes

Notes on what I have built so far, and what comes next.

| File | What it is |
| --- | --- |
| `nn.py` | The `Value` class, plus `Neuron`, `Layer` and `MLP` |
| `test_value.py` | Tests that compare my gradients against PyTorch |
| `train.py` | Trains an MLP on the two-moons dataset |
| `plot_activations.py`, `draw_graph.py`, `draw_network.py` | Pictures of the activations, the computational graph and the network |

## The big idea

A neural network is one long chain of small math operations. To train it, I need to know how much each number in that chain affects the final output. That "how much" is the **gradient**.

Autograd computes gradients automatically. Each value remembers how it was made, so the chain rule can be applied backwards from the output to every input.

## The `Value` class

A `Value` wraps one number and stores:

| Field | What it holds |
| --- | --- |
| `data` | The number itself (forward pass) |
| `grad` | How much the final output changes if this number changes (backward pass) |
| `_prev` | The inputs that produced this value |
| `_op` | The operation that produced it, such as `+` or `*` |
| `_backward` | A function that pushes this value's grad one step back to its inputs |
| `label` | A name, only for debugging and drawing |

## The computational graph

Every operation creates a new `Value` that points back at its inputs through `_prev`. These links form a graph:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b      # c._prev = {a, b}
d = c + a      # d._prev = {c, a}
```

Data flows forward from inputs to output. Gradients flow backward from output to inputs.

## Local gradients (each `_backward`)

Every operation knows only its own slope. The rule is always the same:

> input's grad += (local slope) * (output's grad)

| Operation | Local slope for the input |
| --- | --- |
| `a + b` | 1 for both |
| `a * b` | `b` for a, `a` for b (the other input) |
| `a ** n` | `n * a ** (n - 1)` |
| `tanh(a)` | `1 - out ** 2` |
| `sigmoid(a)` | `out * (1 - out)` |
| `relu(a)` | 1 if `out > 0`, otherwise 0 |

**Why `+=` and not `=`:** a value can be used more than once. In `d = a*b + a`, `a` gets gradient from two places, and they must be added together.

## The graph-wide `backward()`

Each `_backward` only goes one step. `backward()` runs the whole graph:

1. Build a **topological order**: a list where every node comes after its inputs.
2. Set the output's grad to 1.0 (the output changes by 1 when it changes by 1).
3. Walk the list in reverse, calling `_backward()` on each node.

The reverse order matters because a node's grad must be complete before it is passed further back.

Check: for `d = a*b + a` with `a = 2`, `b = 3`, I get `a.grad = 4.0` and `b.grad = 2.0`.

## The other operators

Built from add, multiply and power, so they need no `_backward` of their own:

- `-a` is `a * -1`
- `a - b` is `a + (-b)`
- `a / b` is `a * b ** -1`
- `__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__` handle a plain number on the left, as in `5 * a` or `5 / a`

## Activation functions

A neuron computes `w1*x1 + w2*x2 + b`, which is linear. Stacking linear layers gives another linear function, so without activations a deep network is no better than one layer. An activation bends the output so layers can build up curved shapes.

| Function | Output range | Notes |
| --- | --- | --- |
| ReLU | 0 and up | Keeps positives, zeroes negatives. Cheap. A neuron stuck on negative inputs gets zero gradient and stops learning. |
| Sigmoid | 0 to 1 | S-curve. Gradient peaks at 0.25. |
| tanh | -1 to 1 | S-curve centred on zero. Gradient peaks at 1. Good default for this project. |

**Vanishing gradient:** sigmoid and tanh are nearly flat for large inputs, so their gradient is close to zero there and learning slows down.

## Checking against PyTorch

`test_value.py` builds the same expression twice, once with my `Value` and once with `torch.tensor(..., requires_grad=True)`, calls `backward()` on both, and asserts the output and every input's gradient match. One lambda works for both because my class uses the same operators and method names as PyTorch.

All 23 tests pass. They cover each operation alone, plain numbers on either side, the three activations at negative and positive inputs, a value used twice, a full neuron, and a long chained expression.

Run from the `autograd` folder: `pytest test_value.py -v`. Most of the 20-odd seconds is PyTorch loading.

To do: `__rtruediv__` was added after the tests were written, so add `check(lambda a: 5 / a, 2.0)` to `test_number_on_the_left`.

## The network classes

Three classes, each built from the one before:

| Class | What it holds | What calling it does |
| --- | --- | --- |
| `Neuron(nin)` | `nin` weights and one bias, all `Value`s | `relu(w1*x1 + w2*x2 + ... + b)` |
| `Layer(nin, nout)` | `nout` neurons | Gives every neuron the same inputs, returns their outputs |
| `MLP(nin, nouts)` | One layer per entry in `nouts` | Feeds the output of each layer into the next |

`MLP(2, [16, 16, 1])` means 2 inputs, two hidden layers of 16 neurons, and 1 output. It has 337 parameters.

Details that matter:

- **Weights start random** (between -1 and 1) and **biases start at 0**. If every weight started equal, every neuron in a layer would compute the same thing and learn the same thing.
- **`parameters()`** returns every weight and bias as one flat list. The training loop needs this to update them all.
- **`zero_grad()`** sets every parameter's grad back to 0.
- **The last layer has no ReLU** (`nonlin=False`). ReLU can never output a negative number, so with it the network could never predict -1.
- **A layer with one neuron returns a `Value`, not a list**, so `model(x)` can be used directly.

## Training

Training means repeating four steps:

1. **Forward pass.** Run every data point through the model to get predictions.
2. **Loss.** One number that says how wrong the predictions are. Lower is better.
3. **Backward pass.** `model.zero_grad()`, then `loss.backward()`. Every parameter's grad now says how the loss changes if that parameter goes up.
4. **Update.** `p.data -= lr * p.grad` for every parameter. Moving against the gradient makes the loss go down.

Terms:

- **Mean squared error** is the loss I use: `(prediction - target) ** 2`, averaged over all points.
- **Learning rate (`lr`)** is the step size. Too high and the loss jumps around or blows up. Too low and learning is slow.
- **Accuracy** is the share of points where the prediction has the same sign as the target.
- **Decision boundary** is the line where the model's output crosses zero. One side is predicted +1, the other -1.

### The data

`make_moons` gives 100 points in two interleaving half-moon shapes. A straight line cannot separate them, so it tests whether the network can learn a curve. The labels come as 0 and 1, and I relabel them as -1 and +1.

### Results so far

| Start `lr` | End `lr` | Steps | Final loss | Accuracy | Notes |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.01 | 100 | 0.290 | 94% | Loss spiked to 24 at step 1 |
| 0.05 | 0.02 | 100 | 0.187 | 94% | Loss spiked to 17 at step 2, still falling at the end |
| 0.02 | 0.01 | 200 | | | Current settings in `train.py`, result not recorded yet |

Both finished runs started at loss 1.24 and 50% accuracy (random guessing). The boundary curves the right way around the moons but cuts the tips where they interleave.

A run of 100 steps takes about 4 minutes, because every number is a separate Python object. Real libraries are fast because they do the same math on whole arrays at once.

### Ideas to get past 94%

- Lower the starting learning rate until the early spike is gone.
- Train for more steps.
- Switch to a hinge loss, `(1 - y * pred).relu()`, which only cares that each point is on the correct side by a margin.
- Use a log scale on the loss plot (`ax1.set_yscale('log')`) so the early spike does not flatten the rest of the curve.

## Mistakes I made and fixed

- **Single underscores on operators.** `_add_` does nothing special. Python only uses `__add__`, `__mul__`, `__pow__` (two underscores each side).
- **`__backward` with two leading underscores.** Python renames such methods internally, so it could not be called from outside. It is now `backward`.
- **`self.data.exp()`.** `self.data` is a plain float and floats have no `.exp()`. Use `math.exp(x)` and `math.tanh(x)` with `import math`.
- **`from random import random`.** That imports one function, not the module, so `random.uniform` failed. Use `import random`.
- **ReLU on the output neuron.** The output was clipped at 0 and could never be negative. Passing the same `nonlin` value to every layer did not fix it either; only the last layer should have it turned off.

## Visualization scripts

Run all of these from inside the `autograd` folder with the venv active.

- `python plot_activations.py` plots ReLU, sigmoid and tanh, each with its gradient as a dashed line. The gradients come from my own `backward()`, so a dashed line that does not match the slope of its curve means a bug.
- `python draw_graph.py` draws the computational graph for one neuron. Blue boxes are inputs, orange boxes are results of an operation, and each box shows `data` and `grad`.
- `python draw_network.py` draws an MLP. Circles are neurons showing their bias. Lines are weights: blue is positive, red is negative, thicker is larger. The column headings show each layer's activation.
- `python train.py` ends with two plots: the loss over time, and the decision boundary over the data points.

## Setup

- `python` is not on PATH on this machine. Outside the venv, use `py`.
- Activate the venv from the project root: `.venv\Scripts\Activate.ps1`

## Next steps

Done: `__rtruediv__`, the PyTorch tests, `Neuron` / `Layer` / `MLP`, and a first training run on `make_moons`.

Still to do:

1. Get the two-moons accuracy from 94% to 100% (see the ideas under Training).
2. Add the `5 / a` test to `test_value.py`.
3. Move `Value` into `engine.py`, keeping `nn.py` for the network classes. `engine.py` is still empty.
4. Draw the network before and after training with `draw_network(model)` to see how the weights change.

**Always remember:** zero every grad before each `backward()`. Because of `+=`, gradients from earlier steps pile up otherwise.
