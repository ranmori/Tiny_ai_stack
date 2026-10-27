
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


def plot_live(eval_steps, eval_train, eval_val, learning_rates, target=None):
    # for train.py in a notebook (Kaggle): the loss so far, redrawn at every estimate
    # learning_rates: the learning rate at each of those steps
    # target: the best val loss of an earlier run, drawn as a line to beat
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [3, 1]})

    # left: train and val loss, without step 0 (about 4.4) so the part that matters is not squashed flat
    s, tr, va = eval_steps[1:], eval_train[1:], eval_val[1:]
    ax1.plot(s, tr, 'o-', color='tab:green', markersize=3, label='train loss')
    ax1.plot(s, va, 'o-', color='tab:red', markersize=3, label='val loss')
    if va:
        # mark the best val loss so far, that is the model saved in gpt.pt
        best = min(range(len(va)), key=lambda i: va[i])
        ax1.plot(s[best], va[best], '*', color='black', markersize=14, label=f'best val {va[best]:.4f} (step {s[best]})')
        # the gap between the two lines: the bigger it is, the more the model is memorizing
        ax1.set_title(f'step {eval_steps[-1]}: train {eval_train[-1]:.4f}, val {eval_val[-1]:.4f}, gap {eval_val[-1] - eval_train[-1]:.3f}')
    if target is not None:
        ax1.axhline(target, color='gray', linestyle='--', label=f'best earlier run ({target:.4f})')
    ax1.set_xlabel('step')
    ax1.set_ylabel('cross entropy')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # right: the learning rate, flat unless lr_decay is on
    ax2.plot(eval_steps, learning_rates, 'o-', color='tab:blue', markersize=3)
    ax2.set_title('learning rate')
    ax2.set_xlabel('step')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_attention_training(losses, vocab_size, eval_steps, eval_train, eval_val, wei_before, wei_after, example, baseline=None, save_prefix=None):
    # for the full model (model.py), it opens two windows
    # wei_before, wei_after: the attention weights for one example chunk, before and after training
    #   a list with one entry per block, each entry a list with one (T, T) grid per head
    # example: the characters of that chunk, used to label the rows and columns
    # baseline: the val loss of the bigram, drawn as a line to beat
    # save_prefix: if given, both windows are also saved as png files, save_prefix + '_before.png' and '_after.png'
    n_layer = len(wei_after)
    n_head = len(wei_after[0])
    labels = ['\\n' if ch == '\n' else '_' if ch == ' ' else ch for ch in example]
    shown = ''.join(labels)

    # window 1: the attention before training, one row per block and one column per head
    fig = plt.figure('before training', figsize=(4.2 * n_head, 4 * n_layer))
    grid = fig.add_gridspec(n_layer, n_head)
    for b in range(n_layer):
        for h in range(n_head):
            draw_weights(fig.add_subplot(grid[b, h]), wei_before[b][h], f'block {b + 1} head {h + 1} on "{shown}" before training', labels)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(save_prefix + '_before.png', dpi=80)

    # window 2: the loss on the left over all the rows, then the attention after training
    fig = plt.figure('after training', figsize=(7 + 4.2 * n_head, 4 * n_layer))
    grid = fig.add_gridspec(n_layer, 1 + n_head, width_ratios=[1.6] + [1] * n_head)
    draw_loss(fig.add_subplot(grid[:, 0]), losses, vocab_size, eval_steps, eval_train, eval_val, baseline)
    for b in range(n_layer):
        for h in range(n_head):
            draw_weights(fig.add_subplot(grid[b, 1 + h]), wei_after[b][h], f'block {b + 1} head {h + 1} on "{shown}" after training', labels)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(save_prefix + '_after.png', dpi=80)

    plt.show()


def plot_comparison(results):
    # for compare.py
    # results: one entry per model, with its name, its number of parameters,
    # the steps where the loss was estimated, and the train and val loss at each of those steps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))

    # left: the val loss of every model over the training steps
    for r in results:
        ax1.plot(r['steps'], r['val'], 'o-', markersize=3, label=r['name'])
    ax1.set_title('val loss while training, one more piece switched on in each model')
    ax1.set_xlabel('step')
    ax1.set_ylabel('cross entropy')
    # the first estimate is the untrained model at about 4.2, cut it off so the differences at the end are visible
    lowest = min(min(r['val']) for r in results)
    ax1.set_ylim(lowest - 0.05, lowest + 0.6)
    ax1.legend()

    # right: the final train and val loss of every model side by side
    # a val bar much taller than its train bar means the model is memorizing (overfitting)
    width = 0.38
    xs = range(len(results))
    train_bars = ax2.bar([x - width / 2 for x in xs], [r['train'][-1] for r in results], width, color='tab:green', label='final train loss')
    val_bars = ax2.bar([x + width / 2 for x in xs], [r['val'][-1] for r in results], width, color='tab:red', label='final val loss')
    # write the number on top of every bar
    ax2.bar_label(train_bars, fmt='%.2f', fontsize=8)
    ax2.bar_label(val_bars, fmt='%.2f', fontsize=8)
    ax2.set_xticks(xs, [f"{r['name']}\n{r['params']:,} parameters" for r in results], fontsize=8)
    ax2.set_title('final loss of every model (lower is better)')
    ax2.set_ylabel('cross entropy')
    # start the bars near the lowest value so the differences are visible
    ax2.set_ylim(lowest - 0.3, max(max(r['train'][-1], r['val'][-1]) for r in results) + 0.1)
    ax2.legend()

    plt.tight_layout()
    plt.show()
