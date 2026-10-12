
# the plots for the gpt folder: loss curves, the bigram lookup table, and attention weights
import math
import matplotlib.pyplot as plt


def draw_loss(ax, losses, vocab_size, eval_steps=(), eval_train=(), eval_val=(), baseline=None):
    # the loss going down over the training steps
    # one batch is only 32 chunks so the raw loss is jumpy, the average of the last 50 steps is easier to read
    window = 50
    smooth = [sum(losses[max(0, i - window + 1):i + 1]) / len(losses[max(0, i - window + 1):i + 1]) for i in range(len(losses))]
    ax.plot(losses, alpha=0.3, label='loss of each batch')
    ax.plot(smooth, label=f'average of the last {window} steps')
    # the estimated losses: if the val line climbs away from the train line the model is overfitting
    if len(eval_steps) > 0:
        ax.plot(eval_steps, eval_train, 'o-', color='tab:green', label='estimated train loss')
        ax.plot(eval_steps, eval_val, 'o-', color='tab:red', label='estimated val loss')
    # the loss of a model that knows nothing and gives every character the same probability 1/65
    guessing = math.log(vocab_size)
    ax.axhline(guessing, color='gray', linestyle='--', label=f'pure guessing ({guessing:.2f})')
    # the val loss of an earlier model, the number this one has to beat
    if baseline is not None:
        ax.axhline(baseline, color='black', linestyle=':', label=f'bigram val loss ({baseline:.2f})')
    ax.set_title('loss')
    ax.set_xlabel('step')
    ax.set_ylabel('cross entropy')
    ax.legend()


def draw_weights(ax, wei, title, labels=None):
    # one (T, T) grid of attention weights
    # row = the position that is gathering, column = the position it listens to, darker = more weight
    T = len(wei)
    labels = list(range(T)) if labels is None else labels
    ax.imshow(wei, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('position being listened to')
    ax.set_ylabel('position gathering')
    ax.set_xticks(range(T), labels)
    ax.set_yticks(range(T), labels)
    # write the weight inside every square, the empty upper triangle is the blocked future
    for row in range(T):
        for col in range(row + 1):
            w = wei[row][col].item()
            ax.text(col, row, f'{w:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if w > 0.5 else 'black')


def plot_training(losses, table_before, table_after, char, eval_steps=(), eval_train=(), eval_val=()):
    # for the bigram model (bigram.py)
    # losses: the loss at every training step
    # table_before, table_after: the 65 by 65 lookup table as probabilities, before and after training
    # char: the list of unique characters, used to label the rows and columns
    # eval_steps, eval_train, eval_val: the steps where estimate_loss ran, and the train and val loss it gave
    vocab_size = len(char)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.5))

    # left: the loss
    draw_loss(ax1, losses, vocab_size, eval_steps, eval_train, eval_val)

    # middle and right: the whole model is one 65 by 65 table, so draw it before and after training
    # row = the current character, column = the next character, darker = more likely
    labels = ['\\n' if ch == '\n' else '_' if ch == ' ' else ch for ch in char]
    for ax, table, title in [(ax2, table_before, 'lookup table before training'), (ax3, table_after, 'lookup table after training')]:
        ax.imshow(table, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('next character')
        ax.set_ylabel('current character')
        ax.set_xticks(range(vocab_size), labels, fontsize=5)
        ax.set_yticks(range(vocab_size), labels, fontsize=5)

    plt.tight_layout()
    plt.show()


def plot_attention(wei_average, wei_attention):
    # for the scratch file (attention.py)
    # wei_average: the (T, T) weights of the even average, before attention
    # wei_attention: the (T, T) weights of one chunk with self attention
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_weights(axes[0], wei_average, 'before attention: every earlier position counts the same')
    draw_weights(axes[1], wei_attention, 'with attention: each position chooses its own weights')
    plt.tight_layout()
    plt.show()


def plot_attention_training(losses, vocab_size, eval_steps, eval_train, eval_val, wei_before, wei_after, example, baseline=None):
    # for the attention model (model.py)
    # wei_before, wei_after: one (T, T) grid of attention weights per head, for one example chunk, before and after training
    # example: the characters of that chunk, used to label the rows and columns
    # baseline: the val loss of the bigram, drawn as a line to beat
    n_head = len(wei_before)
    fig = plt.figure(figsize=(7 + 4.6 * n_head, 10))
    # 2 rows: the loss takes the left side over both rows, then one column per head
    grid = fig.add_gridspec(2, 1 + n_head, width_ratios=[1.6] + [1] * n_head)

    # left: the loss
    draw_loss(fig.add_subplot(grid[:, 0]), losses, vocab_size, eval_steps, eval_train, eval_val, baseline)

    # right: who listens to whom in the example chunk, top row before training, bottom row after
    labels = ['\\n' if ch == '\n' else '_' if ch == ' ' else ch for ch in example]
    shown = ''.join(labels)
    for h in range(n_head):
        draw_weights(fig.add_subplot(grid[0, 1 + h]), wei_before[h], f'head {h + 1} on "{shown}" before training', labels)
        draw_weights(fig.add_subplot(grid[1, 1 + h]), wei_after[h], f'head {h + 1} on "{shown}" after training', labels)

    plt.tight_layout()
    plt.show()
