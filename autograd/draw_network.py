
# Draw an MLP: one circle per neuron, one line per weight
import matplotlib.pyplot as plt
from nn import MLP


def draw_network(model):
    # the number of circles in each column: the inputs first, then every layer
    nin = len(model.layers[0].neurons[0].w)
    sizes = [nin] + [len(layer.neurons) for layer in model.layers]

    # where each circle goes: one column per layer, centred vertically on zero
    pos = [[(col, (n - 1) / 2 - row) for row in range(n)] for col, n in enumerate(sizes)]

    fig, ax = plt.subplots(figsize=(2.8 * len(sizes), 1.1 * max(sizes) + 1.5))

    # the biggest weight sets the thickest line
    biggest = max(abs(w.data) for layer in model.layers for n in layer.neurons for w in n.w)

    # draw one line per weight: blue is positive, red is negative, thicker is larger
    for col, layer in enumerate(model.layers):
        for row, neuron in enumerate(layer.neurons):
            for prev_row, w in enumerate(neuron.w):
                (x0, y0), (x1, y1) = pos[col][prev_row], pos[col + 1][row]
                ax.plot([x0, x1], [y0, y1], color='tab:blue' if w.data > 0 else 'tab:red',
                        linewidth=0.3 + 3.5 * abs(w.data) / biggest, alpha=0.7, zorder=1)

    # draw the input circles
    for row, (x, y) in enumerate(pos[0]):
        ax.scatter(x, y, s=1100, color='#cfe8ff', edgecolor='black', zorder=2)
        ax.text(x, y, f'x{row + 1}', ha='center', va='center', fontsize=9, zorder=3)

    # draw the neuron circles, each showing its bias
    for col, layer in enumerate(model.layers):
        for row, neuron in enumerate(layer.neurons):
            x, y = pos[col + 1][row]
            ax.scatter(x, y, s=1100, color='#ffe0b3', edgecolor='black', zorder=2)
            ax.text(x, y, f'b={neuron.b.data:.2f}', ha='center', va='center', fontsize=8, zorder=3)

    # label each column with its size and the activation its neurons use
    top = (max(sizes) - 1) / 2 + 0.9
    ax.text(0, top, f'inputs ({nin})', ha='center', fontsize=10)
    for col, layer in enumerate(model.layers):
        name = 'output' if col == len(model.layers) - 1 else f'hidden {col + 1}'
        activation = 'ReLU' if layer.neurons[0].nonlin else 'linear'
        ax.text(col + 1, top, f'{name} ({len(layer.neurons)})\n{activation}', ha='center', fontsize=10)

    ax.set_title(f'{len(model.parameters())} parameters   |   blue = positive weight, red = negative, thicker = larger',
                 fontsize=10, pad=30)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-(max(sizes) - 1) / 2 - 0.7, top + 0.5)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # 2 inputs -> two hidden layers of 4 neurons -> 1 output
    model = MLP(2, [4, 4, 1])
    draw_network(model)
    plt.show()
