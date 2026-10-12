
# the plots for the bigram model: the loss curve and the lookup table before and after training
import math
import matplotlib.pyplot as plt


def plot_training(losses, table_before, table_after, char, eval_steps=(), eval_train=(), eval_val=()):
    # losses: the loss at every training step
    # table_before, table_after: the 65 by 65 lookup table as probabilities, before and after training
    # char: the list of unique characters, used to label the rows and columns
    # eval_steps, eval_train, eval_val: the steps where estimate_loss ran, and the train and val loss it gave
    vocab_size = len(char)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.5))

    # left: the loss going down over the training steps
    # one batch is only 32 chunks so the raw loss is jumpy, the average of the last 50 steps is easier to read
    window = 50
    smooth = [sum(losses[max(0, i - window + 1):i + 1]) / len(losses[max(0, i - window + 1):i + 1]) for i in range(len(losses))]
    ax1.plot(losses, alpha=0.3, label='loss of each batch')
    ax1.plot(smooth, label=f'average of the last {window} steps')
    # the estimated losses: if the val line climbs away from the train line the model is overfitting
    if len(eval_steps) > 0:
        ax1.plot(eval_steps, eval_train, 'o-', color='tab:green', label='estimated train loss')
        ax1.plot(eval_steps, eval_val, 'o-', color='tab:red', label='estimated val loss')
    # the loss of a model that knows nothing and gives every character the same probability 1/65
    guessing = math.log(vocab_size)
    ax1.axhline(guessing, color='gray', linestyle='--', label=f'pure guessing ({guessing:.2f})')
    ax1.set_title('loss')
    ax1.set_xlabel('step')
    ax1.set_ylabel('cross entropy')
    ax1.legend()

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
