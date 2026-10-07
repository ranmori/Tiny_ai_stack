
# Plot activation functions and their gradients
import matplotlib.pyplot as plt
from nn import Value

# Define the range of input values
xs = [i / 20 for i in range(-120, 121)]   # -6.0 to 6.0
# set up the plot
# make a figure with 1 row and 3 columns of subplots, each for one activation function
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Plot each activation function
for ax, name in zip(axes, ['relu', 'sigmoid', 'tanh']):
    # compute the activation function and its gradient for each input value
    ys, grads = [], []
    for x in xs:
        # create a Value object for the input
        v = Value(x)
        # call the activation function by name using getattr
        out = getattr(v, name)()   # e.g. v.tanh()
        # compute the gradient of the output with respect to the input
        out.backward()
        # store the output value and its gradient
        ys.append(out.data)
        # store the gradient
        grads.append(v.grad)       # slope of the activation at x
   # plot the activation function and its gradient
    ax.plot(xs, ys, label=name)
    ax.plot(xs, grads, linestyle='--', label='gradient')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(name)
    ax.set_xlabel('input')
    ax.legend()

plt.tight_layout()
plt.show()
