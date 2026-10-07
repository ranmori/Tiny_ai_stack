
# Train the MLP to separate two interleaving half-moons
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from nn import MLP



# fix the seed so every run starts from the same weights
# what we are trying here is to show that the MLP can learn to separate the two moons, not to show that it can do so from a random starting point
random.seed(1337)


# the data: 100 points (x1, x2), each belonging to one of two moons
# n_samples: how many points to generate
# y is relabeled to be -1/+1 instead of 0/1, because the MLP is trained to output a number whose sign indicates the class


# noise: how much to jitter the points so that it'isn't a perfect circle (0.0 = no noise, 1.0 = lots of noise)
X, y = make_moons(n_samples=100, noise=0.1, random_state=1)
# to a list of lists, so that each point is a list of two numbers instead of a numpy array
# to a list, so that each label is a number instead of a numpy array

X = X.tolist()
# positive to negative, negative to positive, so that the two moons are flipped vertically
y = [yi * 2 - 1 for yi in y.tolist()]   # relabel 0/1 as -1/+1

# the model: 2 inputs -> two hidden layers of 16 neurons -> 1 output
model = MLP(2, [16, 16, 1])
print(f'number of parameters: {len(model.parameters())}')


def loss_and_accuracy():
    # forward pass: one prediction per data point
    # 
    preds = [model(x) for x in X]
    # mean squared error: how far each prediction is from its target, squared, then averaged
    loss = sum((pred - yi) ** 2 for pred, yi in zip(preds, y)) / len(y)
    # a prediction counts as correct if it has the same sign as its target
    accuracy = sum((pred.data > 0) == (yi > 0) for pred, yi in zip(preds, y)) / len(y)
    return loss, accuracy


# the training loop
steps = 300
losses = []
for step in range(steps):
    # forward pass
    loss, accuracy = loss_and_accuracy()

    # backward pass: zero the old grads first, because backward() accumulates with +=
    model.zero_grad()
    loss.backward()

    # update: nudge every parameter a small step against its gradient
    lr = 0.02 - 0.001 * step / steps   # start at 0.05 and shrink towards 0.02
    for p in model.parameters():
        p.data -= lr * p.grad

    losses.append(loss.data)
    if step % 10 == 0 or step == steps - 1:
        print(f'step {step:3d}   loss {loss.data:.4f}   accuracy {accuracy * 100:.0f}%')


# plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# left: the loss going down over the training steps
ax1.plot(losses)
ax1.set_title('loss')
ax1.set_xlabel('step')
ax1.set_ylabel('mean squared error')

# right: the decision boundary, found by asking the model about every point on a grid
h = 0.1
x1s = [min(x[0] for x in X) - 0.5 + i * h for i in range(int((max(x[0] for x in X) - min(x[0] for x in X) + 1) / h) + 1)]
x2s = [min(x[1] for x in X) - 0.5 + i * h for i in range(int((max(x[1] for x in X) - min(x[1] for x in X) + 1) / h) + 1)]
Z = [[1 if model([x1, x2]).data > 0 else -1 for x1 in x1s] for x2 in x2s]
ax2.contourf(x1s, x2s, Z, levels=[-2, 0, 2], colors=['tab:red', 'tab:blue'], alpha=0.25)
ax2.scatter([x[0] for x in X], [x[1] for x in X],
            c=['tab:blue' if yi > 0 else 'tab:red' for yi in y], edgecolor='black')
ax2.set_title(f'decision boundary (accuracy {accuracy * 100:.0f}%)')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')

plt.tight_layout()
plt.show()
