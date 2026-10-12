# GPT notes

Notes on what I have built so far in the `gpt` folder, and what comes next.

| File | What it is | State |
| --- | --- | --- |
| `data/input.txt` | The training text: about 1.1 million characters of Shakespeare | Ready |
| `train.py` | Loads the text, tokenizes it, makes batches, trains the model, calls the plots | Runs end to end |
| `model.py` | `BigramLanguageModel`, the class only | Done |
| `plots.py` | `plot_training(...)`: the loss curve and the lookup table before and after | Done |
| `generate.py` | Will load saved weights and write text without retraining | Written but commented out |
| `bigram.py` | | Empty |

Run from the project root: `python gpt/train.py`. The plot window opens when training finishes.

## The big idea

A language model predicts the next token given the tokens before it. Train it on enough text and it can write new text one token at a time: predict, pick one, add it to the end, repeat.

In the autograd folder I built the gradient machinery by hand. Here PyTorch does that part, so the work is in the data and the model.

## Step 1: the text

The dataset is one long string. Measured from `data/input.txt`:

- 1,115,393 characters
- 65 unique characters: newline, space, some punctuation, `3`, and upper and lower case letters

## Step 2: tokenizing (text to numbers)

The model cannot read text, only numbers. A **tokenizer** converts between the two. Mine works at the character level: every unique character gets one integer.

```python
char = sorted(list(set(text)))   # the unique characters, sorted
vocab_size = len(char)           # 65
stoi = {ch: i for i, ch in enumerate(char)}   # string to integer
itos = {i: ch for i, ch in enumerate(char)}   # integer to string
encode = lambda s: [stoi[c] for c in s]             # 'hi' -> [46, 47]
decode = lambda l: ''.join([itos[i] for i in l])    # [46, 47] -> 'hi'
```

- `set(text)` removes duplicates, `sorted` gives a fixed order so the mapping is the same every run.
- `decode(encode(s))` must give back `s`. That is the test that the tokenizer works.
- Real GPTs use sub-word tokens (pieces of words) with a vocabulary of tens of thousands. Character level means a tiny vocabulary but long sequences.

## Step 3: the data as a tensor

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

The whole text becomes one long row of integers. `torch.long` means 64-bit integers, which is what an embedding table expects as indexes.

## Step 4: train and validation split

```python
n = int(0.9 * len(data))
train_data = data[:n]   # first 90%
val_data = data[n:]     # last 10%
```

The model only learns from `train_data`. `val_data` is text it has never seen, so the loss on it shows whether the model is learning patterns or just memorizing. If the training loss keeps dropping while the validation loss rises, that is **overfitting**.

## Step 5: context and targets

`block_size` is the longest context the model sees at once (8 for now).

A chunk of `block_size + 1` characters holds `block_size` training examples, because every prefix predicts the next character:

```python
x = train_data[:block_size]        # the inputs
y = train_data[1:block_size + 1]   # the same chunk shifted one to the right
```

- Given `x[:1]`, the target is `y[0]`.
- Given `x[:2]`, the target is `y[1]`.
- And so on up to the full 8.

So `y[t]` is always the character that follows `x[:t+1]`. Training on every length from 1 to `block_size` means the model can later start generating from a single character.

## Step 6: batches

Feeding one chunk at a time is slow, so several chunks are stacked and processed together.

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
```

- `ix` is `batch_size` random starting positions.
- The upper bound `len(data) - block_size` keeps every chunk fully inside the data, including the extra character `y` needs.
- `torch.stack` piles the chunks into one 2D tensor.

The two shape letters used everywhere:

| Letter | Name | Meaning | Value now |
| --- | --- | --- | --- |
| B | batch | How many independent chunks at once | 4 for the printed examples, 32 for training |
| T | time | How many positions in each chunk (`block_size`) | 8 |
| C | channels | How many numbers describe each position (here `vocab_size`) | 65 |

With a batch of 4, `xb` and `yb` are both `(B, T)` = `(4, 8)`. That is 32 separate next-character questions in one batch. The chunks in a batch never see each other.

## Step 7: the bigram model (`model.py`)

The simplest possible language model: predict the next character from **only the current character**. No context at all.

The whole model is one lookup table, `nn.Embedding(vocab_size, vocab_size)`, a 65 by 65 grid of numbers that starts random. Feed in character 7, read off row 7, and those 65 numbers are the scores for what comes next.

- **Logits** are those raw scores, one per possible next character. Higher means more likely. Shape `(B, T, C)`.
- **Softmax** turns logits into probabilities that add up to 1.
- **Cross-entropy** is the loss: it is low when the model gave a high probability to the correct next character.

### `forward(idx, targets=None)`

Looks up the logits. If targets are given, it also computes the loss.

`F.cross_entropy` wants `(questions, options)` and `(answers,)`, so the logits are reshaped from `(B, T, C)` to `(B*T, C)` and the targets from `(B, T)` to `(B*T)`. That is 32 questions, 65 options each, 32 answers.

### `generate(idx, max_new_tokens)`

Repeats this loop:

1. Run the model on everything so far.
2. Keep only the logits for the last position.
3. Softmax them into probabilities.
4. Sample one character with `torch.multinomial` (a weighted random pick, so the output varies).
5. Append it and go again.

### What to expect before training

With 65 characters and a model that knows nothing, a fair guess gives each character probability 1/65. The loss for that is `-ln(1/65)`, about **4.17**. An untrained model should score near that (a little higher, because random logits are not perfectly even), and its generated text is random characters.

Measured: my untrained model scores **4.95** on the first batch, and its text is random characters.

## Step 8: connecting the files

`train.py` starts with `from model import BigramLanguageModel`. Both files are in the same folder, and Python looks in the running script's own folder for imports, so no path is needed.

Importing a file runs everything in it, top to bottom. So `model.py` holds only the imports and the class, and does nothing when imported. The lines that build and use the model live in `train.py`, which has `vocab_size`, `xb`, `yb` and `decode`.

`vocab_size` is handed over as an argument, `BigramLanguageModel(vocab_size)`. `model.py` must never import from `train.py`, or the two files would import each other.

## Step 9: the training loop

The same four steps as the autograd folder, in PyTorch form:

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(1000):
    xb, yb = get_batch('train')            # a fresh random batch
    logits, loss = m(xb, yb)               # forward pass and loss
    optimizer.zero_grad(set_to_none=True)  # clear the old gradients
    loss.backward()                        # backward pass
    optimizer.step()                       # update every parameter
```

| Autograd folder | Here |
| --- | --- |
| `model.zero_grad()` | `optimizer.zero_grad(set_to_none=True)` |
| `loss.backward()` on my own `Value` | `loss.backward()` on a tensor |
| `p.data -= lr * p.grad` in a loop | `optimizer.step()` |

- **The optimizer** owns the update rule. `m.parameters()` hands it every trainable number, here the 65 x 65 = 4,225 entries of the table.
- **AdamW** is smarter than the plain `p.data -= lr * p.grad`. It keeps a running average of each parameter's gradient and squared gradient, so every parameter gets its own step size. The W is weight decay, a gentle pull of the weights towards zero.
- **`lr=1e-3`** is 0.001, the learning rate.
- **Each step uses a new random batch**, not the whole dataset. That makes every step cheap but the loss jumpy from step to step.
- **`loss.item()`** turns a one-number tensor into a plain Python float, for printing or storing.

## Step 10: the plots (`plots.py`)

`plot_training(losses, table_before, table_after, char)` draws three panels:

- **Loss:** every batch's loss, a 50-step average that is easier to read, and a dashed line at 4.17 for pure guessing.
- **Lookup table before training:** the model's whole 65 by 65 table as a grid. Row is the current character, column is the next character, darker is more likely.
- **Lookup table after training:** the same grid after the loop.

`train.py` collects what the plots need, because only the training loop has it:

- `table_before`, a copy of the table as probabilities, taken before the loop. `.detach().clone()` makes a separate copy that training cannot change.
- `losses`, a list that gets `loss.item()` appended every step.
- `table_after`, the table as probabilities after the loop.

## Results so far

| Steps | Batch size | Loss at step 0 | Final loss | Notes |
| --- | --- | --- | --- | --- |
| 1,000 | 32 | 4.70 | 3.58 | Still falling in a straight line at the end |

- The loss only drops below the pure-guessing line (4.17) at about step 450. Before that, the random starting table is worse than guessing.
- The before and after tables look almost the same: scattered random dots. The model has barely learned anything yet.
- 1,000 steps is not enough. With 10,000 the loss should settle around 2.5, and the table should show clear structure.

## Mistakes I made and fixed

- **`train.py` read itself.** It opened `"gpt/train.py"` instead of `"gpt/data/input.txt"`. It was easy to miss because the script happened to contain exactly 65 unique characters too, so `vocab_size` printed 65 either way. The giveaway was the length: about 3,000 characters instead of 1,115,393.
- **`print(text)` on the full file** would print over a million characters. It is now `print(text[:1000])`.
- **`model.py` used names it did not have.** `vocab_size`, `xb`, `yb` and `decode` live in `train.py`, so those lines moved there.
- **Printing the loss on every step** flooded the terminal. It now prints only when `steps % 1000 == 0`.

## Still to fix

- **`data.type` without brackets** prints the method itself, not the type. Use `data.dtype`.
- **`generate.py` is still commented out.** `train.py` now saves `gpt/bigram.pt` with the keys `'model_state_dict'` and `'char'`, which are the names `generate.py` reads, so it only needs uncommenting. The `.pt` file is in `.gitignore`.
- **No text is generated after training.** The only `generate` call is before the loop, so only the random text is shown.
- **`val_data` is never used.** Nothing measures the validation loss yet.

## Next steps

1. Train for 10,000 steps and add the result to the table above.
2. Generate text after the loop and compare it with the untrained text.
3. Uncomment `generate.py` and run it with `python gpt/generate.py`.
4. Add a function that estimates the loss on both train and validation data.
5. Then move past the bigram: self-attention, so each character can look at the ones before it.
