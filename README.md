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

*Text written by a 10.8 million parameter model (val loss 1.30).*

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
| + 10,000 steps instead of 5,000 | 10.8 M | 1.30 |
| + learning rate starting at 6e-4 and decaying | 10.8 M | **1.27 to 1.28** (two runs) |

Two runs with identical settings end about 0.005 apart (the GPU adds numbers in a slightly different order each time), so differences smaller than that mean nothing. Every setting above was chosen by looking at this val loss, so it is a little optimistic. See [Evaluation](#evaluation) for a test set that was never used for any choice.

Some things that were tried and measured, but not kept:

- **More dropout** (0.3 instead of 0.2) shrank the train/val gap but did not lower the val loss.
- **Running all attention heads in one matrix multiply** was only 3% faster on the GPU.
- **Learning rate decay** from the same starting rate made the result worse, because the model was still improving. It only helped together with a higher start.

The biggest single win was **more data**. The model had stopped improving with 1 million characters, so `gpt/data/prepare.py` adds 29 more Shakespeare plays from Project Gutenberg and leaves out the plays the validation text comes from.

## Evaluation

`python gpt/evaluate.py` scores the saved model on two texts, going through each one completely in fixed windows, so the result is the same every run:

| Text | What it is | Used to choose settings? |
| --- | --- | --- |
| Val | The end of *The Taming of the Shrew* and *The Tempest* (111,540 characters) | Yes |
| Test | Shakespeare's poems: the Sonnets, *Venus and Adonis*, *The Rape of Lucrece* and three short ones (265,764 characters) | Never |

Every play is already used for training or val, so the poems are the only Shakespeare the model never saw. They are verse without speakers, a different style from the plays, so the test loss is not comparable with the val loss. Only 0.11% of their 6-word runs appear anywhere in the training text.

| Model | Val | Test (poems) |
| --- | --- | --- |
| Fixed learning rate 3e-4, 10,000 steps | 1.2958 | 1.3566 |

For comparison, the best possible bigram (real character-pair frequencies from the training text) scores 2.46 on val.

## Inference

### KV cache

`generate` used to run the whole context through the model for every new character. The keys and values of earlier characters never change, so they are now kept in a cache and only the newest character goes through the model. `python gpt/benchmark.py` first checks that nothing changed (logits within 1.3e-5, and the same 255 characters for the same seed), then measures on a laptop CPU (4 threads):

| Method | 250 characters | 1,000 characters |
| --- | --- | --- |
| No cache | 16 per second | 10 per second |
| **KV cache** | **60 per second** | **58 per second** |

3.8 times faster for 250 characters and 5.5 times for 1,000. Without the cache it slows down as the context grows; with it the speed stays flat.

### Serving and batching

`gpt/server.py` is a FastAPI server with `/generate` and a streaming `/generate/stream`. `gpt/batcher.py` adds request batching: requests that arrive within 10 ms are generated together in one forward pass (prompts left-padded, with a padding mask and per-row positions). Batched requests give exactly the same text as requests run alone.

`python gpt/loadtest.py`, streaming 200 characters per user, laptop CPU:

| Users at once | Characters/second, no batching | Characters/second, batching | Total time p50, no batching | Total time p50, batching |
| --- | --- | --- | --- | --- |
| 1 | 60 | 58 | 3.3 s | 3.4 s |
| 2 | 82 | 97 | 4.9 s | 4.1 s |
| 4 | 49 | 96 | 16.0 s | 4.5 s |
| 8 | 51 | 159 | 31.3 s | 10.1 s |
| 16 | 40 | **182** | 79.1 s | **11.5 s** |

Without batching the requests' threads fight over the same 4 cores, so the server gets slower overall as users are added. With batching, throughput grows 4.5 times at 16 users. The cost: this is static batching, so a request that misses the 10 ms window waits for the running batch to finish before its first character (4 to 11 s at p95). Continuous batching, where a new request joins a running batch, is the next step.

## Running it

### Quick start

```bash
git clone https://github.com/ranmori/Tiny_ai_stack.git
cd Tiny_ai_stack
pip install -r requirements.txt
python gpt/generate.py
```

The trained model (`gpt.pt`, about 52 MB) is not in git. The first time, `generate.py` downloads it from the GitHub release.

### Server

```bash
uvicorn server:app --app-dir gpt --port 8000
curl -N localhost:8000/generate/stream -H "content-type: application/json" -d '{"prompt": "ROMEO:
", "max_new_tokens": 200}'
```

Or in Docker (CPU-only PyTorch, the model is downloaded on start):

```bash
docker build -t tiny-gpt .
docker run -p 8000:8000 tiny-gpt
```

`BATCHING=0` switches request batching off.

### Scripts

From the project root:

| Command | What it does | Time |
| --- | --- | --- |
| `python gpt/train.py` | Trains the GPT, writes text before and after, saves `gpt/gpt.pt`, shows the loss and attention plots | About 20 min on a CPU |
| `python gpt/generate.py` | Writes 500 characters with the saved model | Seconds |
| `python gpt/evaluate.py` | Val and test loss over the whole texts, the same every run | About 2 min |
| `python gpt/benchmark.py` | Checks the KV cache and measures characters per second | A few minutes |
| `python gpt/loadtest.py` | Load test against a running server | A few minutes |
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
