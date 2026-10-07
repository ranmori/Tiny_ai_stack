
# Draw the computational graph behind a Value, with each node's data and gradient
import matplotlib.pyplot as plt
from nn import Value


def trace(root):
    # walk backwards from the root and collect every node and every edge (child -> parent)
    nodes, edges = [], []
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
                edges.append((child, v))
            nodes.append(v)   # same order as the topological sort in backward()
    build(root)
    return nodes, edges


def layout(nodes, edges):
    # decide which column each node sits in: inputs on the left, the output on the right
    column = {}
    for v in nodes:
        # a node goes one column to the right of its furthest-right input
        column[v] = 1 + max((column[c] for c in v._prev), default=-1)
    for v in nodes:
        # pull each leaf (a node with no inputs) right next to the first node that uses it
        if not v._prev:
            parents = [p for c, p in edges if c is v]
            if parents:
                column[v] = min(column[p] for p in parents) - 1

    # spread the nodes of each column out vertically, centred on zero
    rows = {}
    for v in nodes:
        rows.setdefault(column[v], []).append(v)
    pos = {}
    for col, members in rows.items():
        for i, v in enumerate(members):
            pos[v] = (col, (len(members) - 1) / 2 - i)
    return pos


def draw_graph(root):
    nodes, edges = trace(root)
    pos = layout(nodes, edges)

    n_cols = 1 + max(x for x, y in pos.values())
    n_rows = 1 + max(y for x, y in pos.values()) - min(y for x, y in pos.values())
    fig, ax = plt.subplots(figsize=(2.6 * n_cols + 1, 1.5 * n_rows + 1))

    # draw an arrow from each input to the node it feeds into
    for child, parent in edges:
        ax.annotate('', xy=pos[parent], xytext=pos[child],
                    arrowprops=dict(arrowstyle='->', color='gray', shrinkA=38, shrinkB=38))

    # draw each node as a box showing its name, the operation that made it, its data and its grad
    for v in nodes:
        title = v.label or '(unnamed)'
        if v._op:
            title += f'  [{v._op}]'
        text = f'{title}\ndata = {v.data:.4f}\ngrad = {v.grad:.4f}'
        # inputs are blue, results of an operation are orange
        color = '#ffe0b3' if v._op else '#cfe8ff'
        ax.text(*pos[v], text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black'))

    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # a single neuron with two inputs: out = tanh(x1*w1 + x2*w2 + b)
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.88, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    total = x1w1 + x2w2; total.label = 'sum'
    n = total + b; n.label = 'n'
    out = n.tanh(); out.label = 'out'

    # fill in every grad, then draw
    out.backward()
    draw_graph(out)
    plt.show()
