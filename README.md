# Tiny AI Stack

A small language model built from the ground up, one piece at a time: first a scalar autograd engine, then a character-level GPT that writes Shakespeare-style text.

Every step is measured and written up in the notes, including the experiments that did not work.

```text
LEONTES:
Come, sir,
How now! What say you?
I know you? What say you this?

VOLUMNIA:
I have heard good slander, sir: it is comfort, sir!
```

*Text written by the best model so far (10.8 million parameters, val loss 1.30).*

## What is in here

| Folder | What it is | Notes |
| --- | --- | --- |
| [`autograd/`](autograd/) | A scalar autograd engine (`Value`), a small MLP built on it, tests against PyTorch, and a two-moons classifier | [`autograd/Notes.md`](autograd/Notes.md) |
| [`gpt/`](gpt/) | A character-level GPT in PyTorch: bigram baseline, self-attention, the full transformer, training, text generation and plots | [`gpt/notes.md`](gpt/notes.md) |

## The GPT, step by step

Validation loss (cross entropy, lower is better) on the same held-out text, the end of *The Taming of the Shrew* and *The Tempest*:

| Model | Parameters | Val loss |
| --- | --- | --- |
| Pure guessing (1 of 65 characters) | 0 | 4.17 |
| Bigram (one lookup table) | 4,225 | 2.50 |
| One self-attention head | | 2.39 |
| Four heads | | 2.22 |
| Full transformer: feed-forward, 3 blocks, residual, layer norm | 42,369 | 2.06 |
| Scaled up on a laptop CPU (context 64, size 64) | 161,985 | 1.66 |
| Tutorial size on a GPU (context 256, size 384, 6 blocks) | 10.8 M | 1.48 |
| + 4.5 times more training text | 10.8 M | 1.37 |
| + 10,000 steps instead of 5,000 | 10.8 M | **1.30** |

Some things that were tried and measured, but not kept:

- **More dropout** (0.3 instead of 0.2) shrank the train/val gap but did not lower the val loss.
- **Running all attention heads in one matrix multiply** was only 3% faster on the GPU.
- **Learning rate decay** from the same starting rate made the result worse, because the model was still improving.

The biggest single win was **more data**. The model had stopped improving with 1 million characters, so `gpt/data/prepare.py` adds 29 more Shakespeare plays from Project Gutenberg and leaves out the plays the validation text comes from.

## Running it

Needs Python 3 with PyTorch and matplotlib. The autograd folder also uses scikit-learn and pytest.

```bash
pip install torch matplotlib scikit-learn pytest
```

From the project root:

| Command | What it does | Time |
| --- | --- | --- |
| `python gpt/train.py` | Trains the GPT, writes text before and after, saves `gpt/gpt.pt`, shows the loss and attention plots | About 20 min on a CPU |
| `python gpt/generate.py` | Writes 500 characters with the saved model | Seconds |
| `python gpt/compare.py` | Trains six models, switching on one transformer piece at a time | About 15 min |
| `python gpt/data/prepare.py` | Rebuilds `gpt/data/extra.txt` from Project Gutenberg | Seconds |

From inside `autograd/`:

| Command | What it does |
| --- | --- |
| `pytest test_value.py` | Checks the autograd gradients against PyTorch |
| `python train.py` | Trains the MLP on two moons and plots the decision boundary |

On a CPU, `train.py` uses a small model. On a GPU it switches to the 10.8 M parameter model automatically.

### On a free GPU (Kaggle)

In a Kaggle notebook with a GPU accelerator and internet switched on:

```python
!rm -rf /kaggle/working/Tiny_ai_stack
!git clone https://github.com/ranmori/Tiny_ai_stack.git
%cd /kaggle/working/Tiny_ai_stack
import sys
sys.path.insert(0, 'gpt')
%run gpt/train.py
```

It should print `device: cuda`. A live loss graph redraws under the cell every 500 steps, and the best model is saved to `gpt/gpt.pt` as it trains. 10,000 steps take about 1 hour 50 minutes.

## Credits

- Built while following Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) lessons (micrograd and "Let's build GPT").
- `gpt/data/input.txt` is the Tiny Shakespeare dataset. `gpt/data/extra.txt` is made from [Project Gutenberg's Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100), which is in the public domain.
