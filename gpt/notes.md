# GPT notes

Notes on what I have built so far in the `gpt` folder, and what comes next.

| File | What it is | State |
| --- | --- | --- |
| `data/input.txt` | The training text: about 1.1 million characters of Shakespeare | Ready |
| `train.py` | Loads the text, tokenizes it, makes batches, trains the full model, estimates the loss, writes text, saves the model, calls the plots | Runs end to end |
| `model.py` | `Head`, `MultiHeadAttention`, `FeedForward`, `Block` and `GPTLanguageModel` | A complete small GPT |
| `compare.py` | Trains six models, switching on one more piece each time, and plots what each piece adds | Done |
| `generate.py` | Loads the model saved by `train.py` and writes text without retraining | Updated for the full model |
| `data/prepare.py` | Downloads the complete works and writes `data/extra.txt`, 29 more plays cleaned to look like `input.txt` | Done |
| `bigram.py` | `BigramLanguageModel`, the baseline to beat | Done, kept for reference |
| `plots.py` | One plot function per stage, plus two shared helpers | Done |
| `attention.py` | Scratch file for learning attention, no model in it | Steps 1 and 2 done |
| `gpt.pt`, `attention.pt`, `bigram.pt` | Trained models saved by `train.py` at each stage | Not on GitHub, `*.pt` is in `.gitignore` |

All of these run from the project root:

| Command | What happens | How long |
| --- | --- | --- |
| `python gpt/train.py` | Trains the full model, then opens two plot windows | About 20 minutes |
| `python gpt/compare.py` | Trains six models, then opens the comparison plot | About 15 minutes |
| `python gpt/generate.py` | Writes 500 characters with the saved model | Seconds |
| `python gpt/attention.py` | The scratch file and its plot | Seconds |

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

## Step 7: the bigram model (`bigram.py`)

Steps 7 to 12 describe the bigram stage. The class now lives in `bigram.py`, and `train.py` has moved on to the attention model, but the training loop, `estimate_loss()` and the saving work the same way.

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

for steps in range(10000):
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

`plot_training(losses, table_before, table_after, char, eval_steps, eval_train, eval_val)` draws three panels:

- **Loss:** every batch's loss, a 50-step average that is easier to read, the estimated train and validation loss as dots, and a dashed line at 4.17 for pure guessing.
- **Lookup table before training:** the model's whole 65 by 65 table as a grid. Row is the current character, column is the next character, darker is more likely.
- **Lookup table after training:** the same grid after the loop.

`train.py` collects what the plots need, because only the training loop has it:

- `table_before`, a copy of the table as probabilities, taken before the loop. `.detach().clone()` makes a separate copy that training cannot change.
- `losses`, a list that gets `loss.item()` appended every step.
- `eval_steps`, `eval_train`, `eval_val`, three lists filled every time `estimate_loss()` runs.
- `table_after`, the table as probabilities after the loop.

The lookup table panels only make sense for the bigram, whose whole model is one table. The attention model has its own plot, described further down.

`plots.py` now has four plot functions and two helpers they share:

| Function | Used by | What it draws |
| --- | --- | --- |
| `plot_training` | The bigram stage | Loss, and the lookup table before and after |
| `plot_attention` | `attention.py` | The even average next to one attention grid |
| `plot_attention_training` | `train.py` now | Two windows: every block's and head's weights before training, and the loss next to the same grids after training |
| `plot_comparison` | `compare.py` | The validation loss of six models while training, and their final train and val loss as bars |
| `draw_loss` | Helper | The loss panel, with an optional baseline line |
| `draw_weights` | Helper | One grid of attention weights with the numbers written in |

## Step 11: estimating the loss

The loss of one batch is jumpy, because it depends on which 32 chunks were picked. `estimate_loss()` averages the loss over 200 batches, once for `train_data` and once for `val_data`, and returns both.

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out
```

- **`@torch.no_grad()`** tells PyTorch not to remember anything for `backward()`. This is only measuring, so it is faster and uses less memory.
- **`m.eval()` and `m.train()`** switch the model between measuring mode and training mode. It makes no difference for the bigram, but later layers such as dropout behave differently in each.
- **Train loss against validation loss** is the overfitting check. If train keeps falling while validation rises, the model is memorizing the training text.

It runs every 1,000 steps inside the loop, and once more after it.

## Step 12: saving the model and generating

`train.py` writes text twice from the same starting character (a single newline, index 0): once before training and once after, each under a heading.

It then saves the model, before the plots, because the plot window blocks the script until it is closed:

```python
torch.save({'model_state_dict': m.state_dict(), 'char': char}, 'gpt/bigram.pt')
```

- `state_dict()` is every trained number in the model.
- `char` is saved too, because `generate.py` needs it to rebuild the same `decode`.

`generate.py` rebuilds a model of the same shape, loads those numbers into it with `load_state_dict`, and generates.

The attention model is saved the same way, to `gpt/attention.pt`.

## Results: the bigram

| Steps | Batch size | Train loss | Val loss | Notes |
| --- | --- | --- | --- | --- |
| 1,000 | 32 | 3.58 (one batch) | | Still falling in a straight line at the end |
| 10,000 | 32 | 2.47 | 2.48 to 2.50 | Flat from about step 7,000 |

The validation number moves a little between runs because each estimate uses 200 random batches.

- **Starting point:** about 4.65 at step 0, worse than the pure-guessing line of 4.17. The loss crosses below that line at about step 450.
- **It levels off at about 2.5.** That is as good as a bigram gets, and it is the number the attention model has to beat.
- **No overfitting.** Train and validation loss stay within about 0.03 of each other.

Text before training:

```text
lfJeukRuaRJKXAYtXzfJ:HEPiu--sDioi;ILCo3pHNTmDwJsfheKRxZCFs
```

Text after 10,000 steps:

```text
Wirsis y GHaitomyou spowhemoule o, an p't ad, s, hethipe, TW:
To ices CAngre, tea yolelarero paint Coth whtu hero t w te
```

After training there are word-length chunks, sensible line breaks, and capitalised names ending in a colon like a play script. It is not English, because each character is chosen by looking at only the one before it.

What the trained lookup table shows:

- Punctuation such as `,` `:` `;` and `?` is almost always followed by a space or a newline.
- `q` is nearly always followed by `u`.
- Most characters are usually followed by a common lowercase letter.
- Capital letters rarely follow anything except a space or newline.

## Attention (`attention.py`)

The bigram's limit is that each character predicts the next using only itself. To use context, each position has to gather information from the positions before it, and never from the ones after, because those are the answers.

### Step 1: averaging the past

The crudest way to gather the past is to average it. This is the version attention builds on:

```python
tril = torch.tril(torch.ones(T, T))              # lower triangle of ones
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float('-inf'))  # block the future
wei = F.softmax(wei, dim=-1)                     # each row becomes an even average
out = wei @ x                                    # (T, T) @ (B, T, C) -> (B, T, C)
```

- **`tril`** is a grid with ones on and below the diagonal. Row `t` has ones in columns 0 to `t`: the positions that row is allowed to see.
- **`masked_fill(..., -inf)`** puts minus infinity wherever `tril` is 0, which is every future position.
- **Softmax** turns each row into weights that add up to 1. Minus infinity becomes exactly 0, and the equal zeros that remain become equal shares.
- **`wei @ x`** makes each position a weighted sum of the positions it can see.

The result: row 0 is `1.0`, row 1 is `0.5, 0.5`, row 3 is `0.25` four times. Every earlier character counts the same.

### Step 2: a single self-attention head

An even average throws away too much. Attention lets each character decide how much to listen to each earlier character. The only change from step 1 is that `wei` starts as computed scores instead of zeros.

Every character produces three vectors through three small linear layers (no bias):

| Vector | Question it answers |
| --- | --- |
| query | What am I looking for? |
| key | What do I contain? |
| value | What do I hand over if someone listens to me? |

```python
k = key(x)     # (B, T, 16)
q = query(x)   # (B, T, 16)
wei = q @ k.transpose(-2, -1) * head_size**-0.5   # (B, T, T) scores

wei = wei.masked_fill(tril == 0, float('-inf'))   # same as step 1
wei = F.softmax(wei, dim=-1)                      # same as step 1

v = value(x)   # (B, T, 16)
out = wei @ v  # (B, T, 16)
```

- **The score** between two positions is the query of one multiplied by the key of the other. It is high when what one is looking for matches what the other contains.
- **`k.transpose(-2, -1)`** swaps the last two dimensions so the shapes line up: `(B, T, 16) @ (B, 16, T)` gives `(B, T, T)`, one score for every pair of positions.
- **`* head_size**-0.5`** divides the scores by the square root of `head_size`. Large scores would make softmax put nearly everything on one character.
- **The output collects `v`, not `x`.** Each position ends up with 16 numbers, a weighted mix of earlier characters' values.
- **`head_size`** is how many numbers the query, key and value vectors have (16 here).

Differences from step 1:

- The weights are uneven. Row 3 came out as `0.32, 0.22, 0.24, 0.21` instead of `0.25` four times.
- Every chunk in the batch gets its own grid of weights, because the scores depend on the characters. In step 1 all chunks shared one fixed grid.
- The upper-right triangle is still all zeros, and every row still adds up to 1.

In this scratch file the weights mean nothing yet: `x` is random and the three layers are untrained. Training is what makes a query learn to find useful keys.

It is called **self**-attention because the queries, keys and values all come from the same `x`.

### Step 3: wiring one head into the model (`model.py`)

`Head` is step 2 as an `nn.Module`, so its three linear layers get trained. Two things are new in it:

- **`register_buffer('tril', ...)`** stores the triangle on the module without making it a trainable parameter. The optimizer never touches it.
- **`self.tril[:T, :T]`** cuts the triangle down to the current length, because while generating the context can be shorter than `block_size`.

The model (called `AttentionLanguageModel` at this stage, now `GPTLanguageModel`) is the bigram with four changes:

| Change | Why |
| --- | --- |
| `n_embd = 32`: the token table gives each character 32 numbers | The bigram table went straight to 65 logits, leaving nothing for attention to work on |
| A position embedding table, one vector per position 0 to 7 | Attention only mixes vectors and has no sense of order. This tells it where a character sits |
| The head, applied to `tok_emb + pos_emb` | Each position gathers from the ones before it |
| `lm_head`, a linear layer from 32 numbers to 65 logits | Turns the result back into one score per character |

```python
tok_emb = self.token_embedding_table(idx)                 # (B, T, n_embd) what each character is
pos_emb = self.position_embedding_table(torch.arange(T))  # (T, n_embd) where it sits
x = tok_emb + pos_emb                                     # (B, T, n_embd)
x = self.sa_heads(x)                                      # (B, T, n_embd)
logits = self.lm_head(x)                                  # (B, T, vocab_size)
```

- **`tok_emb + pos_emb`** works by broadcasting: the `(T, n_embd)` position vectors are added to every chunk in the batch.
- **`generate` crops the context** with `idx[:, -self.block_size:]`. The position table only has `block_size` rows, so a longer context would crash.

### Step 4: multiple heads

One head can only pay attention in one way. `MultiHeadAttention` runs several heads side by side on the same input, so each can look for something different, then glues their outputs together.

```python
self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
...
return torch.cat([h(x) for h in self.heads], dim=-1)
```

- **4 heads of 8 numbers each** (`n_embd // n_head`) are joined by `torch.cat` into 32 numbers per position, the same size as one big head. It costs about the same and works better.
- **`nn.ModuleList`** is a Python list that PyTorch can see inside. With a plain list, the heads' parameters would not be found by `m.parameters()` and would never be trained.
- **`n_embd` must divide evenly by `n_head`.**

### The attention plot

`train.py` takes one fixed chunk of text, `First Ci`, and records every head's weights on it before and after training. `plot_attention_training` draws the loss on the left with the bigram's validation loss as a dotted line, then one column per head: before training on top, after training below.

What the trained heads show on `First Ci`:

- **Before training** every head's weights are spread out and meaningless.
- **Head 3** learned to look almost only at the current character (0.93 to 1.00 down the diagonal).
- **Head 1** mostly looks at the current character, but `r` looks at the `i` before it (0.86) and `C` looks at the space before it (0.95).
- **Head 2** looks back more: `i` at `F` (0.83), `r` at `i` (0.95), `t` at `s` (0.97).
- **Head 4** is a mix, and near the end of the chunk it looks back at the space.

So the heads really did specialise, which a single head could not do.

## Results: attention

10,000 steps, batch size 32, `block_size` 8, `n_embd` 32, learning rate 0.001.

| Model | Train loss | Val loss |
| --- | --- | --- |
| Bigram | 2.47 | 2.50 |
| One head (head size 32) | 2.35 | 2.39 |
| Four heads (head size 8) | 2.16 | 2.22 |

- **Each step helped.** One head beat the bigram by about 0.11, and four heads beat one head by about 0.17.
- **The single head learned very little context.** Its weights sat mostly on the diagonal: look at the current character, plus a little of the one before. That is barely more than a bigram.
- **Train and validation are starting to separate** with four heads (2.16 against 2.22). It is a small gap, but worth watching as the model grows.
- **The four-head loss was still falling** at step 10,000.

Text from the four-head model:

```text
Me, shor, the that lience t'th be.

KIO:
An:
More mavin dige nothame bak, mit?

Tut din or shallk?
```

Real short words now appear (`the`, `that`, `be`, `or`, `More`).

## The five pieces that make it a transformer (`model.py`)

Multi-head attention is the core, but five more pieces turn it into the real thing. The data flows like this:

```text
characters
  -> token embedding + position embedding
  -> Block 1 -> Block 2 -> Block 3
  -> final layer norm
  -> lm_head
  -> logits (65 scores for the next character)
```

and inside every block:

```python
x = x + self.sa(self.ln1(x))     # talk: layer norm, attention, added on top of x
x = x + self.ffwd(self.ln2(x))   # think: layer norm, feed forward, added on top of x
```

### Piece 1: the feed-forward layer

Attention is the positions talking to each other and gathering information. Feed-forward is each position thinking by itself about what it gathered.

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # grow to 4 times the size
    nn.ReLU(),                       # the activation
    nn.Linear(4 * n_embd, n_embd),   # shrink back
    nn.Dropout(dropout),
)
```

- It is the same kind of MLP as in the autograd folder: linear, activation, linear.
- It is applied to every position separately. Positions do not see each other here.
- The middle is 4 times wider to give it room to work, then it shrinks back so the output has the same size as the input.
- `nn.Sequential` runs layers one after another.

### Piece 2: blocks

A `Block` is one round of "talk, then think": multi-head attention followed by feed-forward. Blocks are stacked (`n_layer = 3`), so later blocks work with what earlier blocks figured out.

```python
self.blocks = nn.Sequential(*[Block(...) for _ in range(n_layer)])
```

The `*` unpacks the list, because `nn.Sequential` wants the blocks as separate arguments.

### Piece 3: residual connections

A block does not replace `x`. It adds a correction on top of it: `x = x + ...`.

- `+` passes the gradient straight through, exactly as in my autograd engine, where `__add__` hands `out.grad` to both inputs unchanged.
- So during `backward()` even the first block gets a clear signal. Without this, a stack of blocks trains badly.
- It also means every block starts from the full information and only has to learn what to add.

`MultiHeadAttention` gained a `proj` linear layer for this. It mixes what the separate heads found before the result is added back onto `x`.

### Piece 4: layer norm

`nn.LayerNorm(n_embd)` rescales the 32 numbers of each position so they have mean 0 and spread 1. It has two small sets of trainable numbers, so the model can undo it where it wants to.

- It is applied before attention and before feed-forward in every block, plus once at the very end (`ln_f`).
- The numbers stay in a steady range however deep the stack is, which keeps training stable.

### Piece 5: dropout

`nn.Dropout(0.2)` sets a random 20% of the numbers passing through it to zero, a different set on every step, and only while training.

- The model cannot rely on any single path, so it memorizes less. It is a tool against overfitting.
- It sits in three places: on the attention weights, after `proj`, and at the end of feed-forward.
- **`m.eval()` switches it off and `m.train()` switches it back on.** This is why `estimate_loss()`, the text generation and the attention plot all call `m.eval()` first. Measuring or writing with dropout on would use a damaged model.
- A visible side effect in the loss plot: the per-batch training loss (dropout on) sits above the estimated train loss (dropout off).

### The switches

`Block` and `GPTLanguageModel` take `feed_forward`, `residual` and `layer_norm` arguments that default to `True`. `train.py` never touches them. They exist so that `compare.py` can switch pieces off and measure what each one adds.

## Results: what each piece adds (`compare.py`)

Six models, each with one more piece switched on than the one before. 5,000 steps each, same seed, batch size 32, `block_size` 8, `n_embd` 32, 4 heads.

| Model | Parameters | Train loss | Val loss | Gap |
| --- | --- | --- | --- | --- |
| 4 heads only | 8,609 | 2.24 | 2.28 | 0.04 |
| + feed forward | 16,961 | 2.16 | 2.17 | 0.01 |
| + 3 stacked blocks | 41,921 | 2.24 | 2.28 | 0.04 |
| + residual | 41,921 | 1.99 | 2.09 | 0.10 |
| + layer norm | 42,369 | 1.98 | 2.07 | 0.09 |
| + dropout 0.2 | 42,369 | 2.08 | 2.14 | 0.05 |

What the comparison shows:

- **Feed-forward helps** (2.28 to 2.17). Giving each position a chance to think is worth about 0.11.
- **Stacking blocks without residual connections makes things worse** (2.17 back up to 2.28), even though the model has more than twice the parameters. It also learns slowest of all at the start. A deeper model is harder to train, and this is the problem residual connections exist to solve.
- **Residual connections are the biggest single win** (2.28 to 2.09) with zero extra parameters. The same three blocks suddenly train well.
- **Layer norm helps a little** (2.09 to 2.07). At three blocks the model is not deep enough to need it much.
- **Dropout makes the loss worse at this size** (2.07 to 2.14), but it nearly halves the gap between train and validation (0.09 to 0.05). That is dropout doing its job: less memorizing. This model is too small to overfit badly, so the cost outweighs the benefit. Dropout pays off on bigger models and longer training.
- **More parameters is not the same as better.** Models 3 and 4 have exactly the same parameters and differ by 0.19.

The plot from `compare.py` has the six validation curves on the left and the final train and val loss of each model as bars on the right.

## Results: the full model

`train.py` with everything on: 3 blocks, 4 heads, `n_embd` 32, `block_size` 8, dropout 0.2, 10,000 steps. 42,369 parameters.

| Model | Train loss | Val loss |
| --- | --- | --- |
| Bigram | 2.47 | 2.50 |
| One head | 2.35 | 2.39 |
| Four heads | 2.16 | 2.22 |
| Full model | 2.01 | 2.06 |

The loss was still falling at step 10,000.

Text from the full model:

```text
LING RICHAGYST:
The seak ary, sir you doth slecome.

FORCARIO:
Whind is lool hand nout for esor pake of have
st what Claze quicarss.
```

It now writes in the shape of a play: a name in capitals with a colon, then lines of speech with real words (`sir`, `you`, `doth`, `hand`, `for`, `of`, `have`, `what`).

### What the attention looks like with three blocks

The plot has one row per block and one column per head, on the chunk `First Ci`.

- **Block 1 mostly looks at the character just before.** In head 1 nearly all the weight sits one step left of the diagonal (0.90 to 1.00), and head 2 does mostly the same. With residual connections the current character is already in `x`, so attention no longer has to spend its budget on it.
- **Blocks 2 and 3 are more spread out.** They look two or three characters back, at the start of the word, or at the space. They are working with what block 1 already gathered.
- **Before training** all twelve grids are meaningless smears.

## Scaling up

The model works, but it is tiny and only sees 8 characters at a time. To make it bigger without guessing, I changed one setting at a time and trained each version for 5,000 steps. All runs used the same seed, batch size 32, 4 heads, learning rate 0.001 and no dropout unless the table says otherwise.

This machine has no GPU, so every run is on the CPU. That is why I measured the time per step before anything else.

| Model | Parameters | Train loss | Val loss | Gap |
| --- | --- | --- | --- | --- |
| Context 8, size 32, 3 blocks (the old settings) | 42,369 | 1.97 | 2.06 | 0.08 |
| Context 32 | 43,137 | 1.84 | 1.96 | 0.13 |
| Context 32, size 64 | 159,937 | 1.62 | 1.77 | 0.15 |
| Context 32, size 64, 4 blocks | 209,729 | 1.58 | 1.77 | 0.19 |
| The same, learning rate 3e-4 | 209,729 | 1.74 | 1.89 | 0.15 |
| Context 32, size 64, dropout 0.1 | 159,937 | 1.69 | 1.84 | 0.15 |
| Context 32, size 64, dropout 0.2 | 159,937 | 1.74 | 1.89 | 0.15 |
| **Context 64, size 64** | 161,985 | 1.55 | **1.73** | 0.18 |

"Context" is `block_size` and "size" is `n_embd`.

What it shows:

- **A longer context helps** (2.06 to 1.96 at size 32, and 1.77 to 1.73 at size 64). Seeing more characters gives attention more to work with. It costs almost no parameters, only a bigger position table, but every step is slower because attention compares every pair of positions.
- **A bigger `n_embd` is the biggest win** (1.96 to 1.77). Four times the parameters, but each step is only a little slower.
- **A 4th block is not worth it** (1.773 to 1.768). It costs 30% more time per step for almost nothing.
- **The lower learning rate 3e-4 is worse** at 5,000 steps (1.89 against 1.77). It learns more carefully but too slowly for a run this short. It would make sense for a much longer run, so 0.001 stays.
- **Dropout still makes it worse** (1.77 to 1.84 to 1.89), and this time it does not even close the gap: the gap stays 0.15 with or without it. The model is still too small and trained too briefly to memorize much, so there is nothing for dropout to fix yet. It stays at 0.
- **The gap between train and val is growing** with size (0.08 to 0.18). Something to keep watching if the model gets bigger.

On this CPU one training step takes about 60 ms with the old settings, 100 ms with context 32 and size 64, and 150 ms with context 64 and size 64.

`train.py` now uses context 64, size 64, 3 blocks, 4 heads, no dropout, 10,000 steps. 161,985 parameters, and it trains in about 20 minutes.

| Model | Train loss | Val loss |
| --- | --- | --- |
| Bigram | 2.47 | 2.50 |
| Full model, old settings | 2.01 | 2.06 |
| Full model, scaled up | 1.46 | 1.66 |

The val loss was still falling slowly at the end (1.69 at step 7,000, 1.66 at step 10,000). The gap between train and val grew to 0.20, right at the point where dropout might start to pay off.

Text from the scaled-up model:

```text
ANGEenes, you are out, in very too:
Farewell, unscul the mystenge there,
But wherefore must despile of the gaught,
Which flore-way upon upon the stam,

POMPEY:
I had, you his device did know his fear'd lead,
Meach you weldst it the duned lord! thou rash Orace.

CLIFFORD:
Hence M
```

Whole lines now almost scan as English (`Farewell`, `But wherefore must`, `I thought by how much`), and names from the plays appear (Pompey, Clifford).

The first part of `train.py` still uses a context of 8 for the printed examples, and the attention plot uses only the first 8 characters, because a 64 by 64 grid with the numbers written in would be unreadable.

## Training on a GPU (Kaggle)

The next size up is too slow for this CPU, so it trains on a free GPU instead. Kaggle gives about 30 hours of GPU time a week.

### What changed in the code

A **device** is where the numbers live and the math runs: `'cuda'` is a GPU, `'cpu'` is the processor. A tensor on the GPU and a tensor on the CPU cannot be used in the same calculation, so everything has to be moved to the same place.

| Where | Change |
| --- | --- |
| `train.py` | `device = 'cuda' if torch.cuda.is_available() else 'cpu'` |
| `train.py` | `m = m.to(device)` moves every parameter of the model |
| `train.py` | `get_batch` returns `x.to(device), y.to(device)` |
| `train.py` | The example chunk and the starting context are made on the device |
| `train.py` | The attention weights come back with `.cpu()`, because matplotlib only works with numbers on the CPU |
| `model.py` | `torch.arange(T, device=idx.device)` makes the positions on the same device as the input |
| `generate.py` | `torch.load(..., map_location='cpu')` so a model trained on the GPU also loads on this laptop |

The same `train.py` runs in both places. On the CPU it uses the settings from "Scaling up". On a GPU it switches to the size from the original tutorial:

| Setting | CPU | GPU |
| --- | --- | --- |
| `block_size` | 64 | 256 |
| `n_embd` | 64 | 384 |
| `n_head` | 4 | 6 |
| `n_layer` | 3 | 6 |
| `dropout` | 0 | 0.2 (0.3 was tried, see "Dropout 0.3 on the GPU") |
| Batch size | 32 | 64 |
| Learning rate | 1e-3 | 3e-4 |
| Steps | 10,000 | 10,000 (5,000 before the extra text) |
| Parameters | 161,985 | about 10.8 million |

The tutorial reached a val loss of about 1.48 with these settings.

### How to run it

1. On kaggle.com: **Create, New Notebook**.
2. In the right panel, **Session options, Accelerator**: pick **GPU T4 x2** or **GPU P100**. The phone number on the account has to be verified first.
3. In the same panel, switch **Internet on**, so the notebook can download the code.
4. In the first cell:

```python
!git clone https://github.com/ranmori/Tiny_ai_stack.git
%cd Tiny_ai_stack
import sys
sys.path.insert(0, 'gpt')
%run gpt/train.py
```

- `%cd` moves into the repo, because `train.py` opens `gpt/data/input.txt` from there.
- `sys.path.insert` lets `from model import ...` find the files in `gpt`.
- `%run` runs the script inside the notebook, so the plots appear under the cell.
- It should print `device: cuda`. If it says `cpu`, the accelerator is not switched on.

When it finishes, `gpt.pt` is in the file panel under `/kaggle/working/Tiny_ai_stack/gpt/`. Download it and put it in this laptop's `gpt` folder, and `python gpt/generate.py` writes with it.

### Watching a run on Kaggle

In a notebook, `train.py` shows a **live loss graph** under the cell that redraws at every estimate (every 500 steps on the GPU):

- **Left:** train and val loss, with a star on the best val loss so far (the model saved in `gpt.pt`) and a dashed line at the best val loss of the earlier run with the same settings (1.2970 on the GPU). The title shows the latest losses and the gap between them.
- **Right:** the learning rate, a flat line unless `lr_decay` is on.
- Step 0 (about 4.4) is left out, so the part that matters is not squashed flat.

How it works:

- `get_ipython()` only exists inside a notebook. `'IPKernelApp' in get_ipython().config` is true in a Jupyter kernel like Kaggle's, so in a terminal nothing changes.
- `display(fig, display_id=True)` shows the graph and gives back a handle. `handle.update(fig)` swaps the picture in place instead of adding a new graph every 500 steps.
- `plt.close(fig)` after each update, otherwise the notebook would show every old version at the end of the cell.

At the end, three pictures are saved next to `gpt.pt`, so they can be downloaded from Kaggle's file panel: `plot_loss.png` (the final live graph), `plot_before.png` and `plot_after.png` (the attention plots). `.gitignore` keeps them out of GitHub.

After a code change here, push it to GitHub. The notebook clones the code fresh, so a new session gets the latest version.

The browser saves `gpt.pt` as `gpt.zip`, because a `.pt` file is a zip archive inside. Do not unzip it. Rename it to `gpt.pt` and put it in the `gpt` folder. The model trained on this laptop is kept next to it as `gpt_cpu.pt`.

### Results: the GPU model

10,788,929 parameters, 5,000 steps, about 53 minutes on Kaggle (about 0.63 seconds per step).

| Step | Train loss | Val loss |
| --- | --- | --- |
| 0 | 4.37 | 4.36 |
| 1,000 | 1.54 | 1.73 |
| 2,000 | 1.32 | 1.56 |
| 3,000 | 1.21 | 1.50 |
| 3,500 | 1.17 | 1.48 |
| 4,000 | 1.13 | 1.48 |
| 4,500 | 1.11 | 1.49 |
| 5,000 (final) | 1.07 | 1.49 |

| Model | Parameters | Train loss | Val loss |
| --- | --- | --- | --- |
| Bigram | 4,225 | 2.47 | 2.50 |
| Full model, old settings | 42,369 | 2.01 | 2.06 |
| Full model, scaled up on the CPU | 161,985 | 1.46 | 1.66 |
| **Full model on the GPU** | 10,788,929 | 1.07 | **1.49** |

- **It reached the tutorial's number** (about 1.48).
- **This model overfits.** From step 3,500 the val loss stops at 1.48 while the train loss keeps falling, from 1.17 to 1.07. The gap ends at 0.42, much bigger than anything before. The last 1,500 steps only taught it to memorize the training text. Even with dropout 0.2, 10.8 million parameters is a lot for 1 million characters.
- So 3,500 steps would have given the same model in about two thirds of the time. Stopping when the val loss stops improving is called **early stopping**.

Text from the GPU model:

```text
GLOUCESTER:
I would follow you not took in me;
Let me from heard them be the other.

HASTINGS:
You that stay thou remember'd your meanest.
Go you to, xamillo; I am sorry Aumerle,
My great that you have been to still.

LADY ANNE:
Go to; mercy joy to look you, for me.
```

Most words are real now, the lines have the rhythm of verse, and the speakers belong together: Gloucester, Hastings and Lady Anne are all in Richard III. The sentences still do not make sense. A character-level model of this size learns spelling, words and the look of a play, but not meaning.

`generate.py` runs this model on the laptop's CPU. It takes about half a minute for 500 characters.

## Early stopping (`train.py`)

The GPU model stopped improving at step 3,500 but trained to 5,000, and the model saved at the end was the overfitted one. `train.py` now keeps the best version instead:

```python
if est['val'] < best_val:
    best_val = est['val']
    best_step = steps
    best_state = {k: v.clone() for k, v in m.state_dict().items()}
    save_model(best_state)
elif steps - best_step >= patience * eval_interval:
    break
```

- **Every estimate checks for a new best val loss.** If there is one, the model is copied and saved to `gpt/gpt.pt` straight away. Saving at once means a crash or a Kaggle timeout later still leaves the best model on disk.
- **`.clone()`** makes a separate copy of every tensor. `state_dict()` on its own gives the model's live numbers, which the next training steps would keep changing.
- **`patience = 3`**: if 3 estimates in a row bring no new best, training stops. That is 3,000 steps on the CPU and 1,500 on the GPU. One bad estimate is not enough, because the val estimate wobbles by about 0.01 between runs.
- **After the loop, `m.load_state_dict(best_state)`** puts the best numbers back into the model, so the text and the attention plot at the end come from the same model `generate.py` will load.
- The loop now runs to `max_steps + 1`, so the last estimate happens inside the loop and can also count as a new best.

On Kaggle, the second GPU run with early stopping saved the model at step 4,000 (val loss 1.4825) and kept training to the end. Step 5,000 was only the second estimate without a new best, one short of the `patience` of 3. With the same seed the run was almost identical to the first one, so this is the first run's step-4,000 model. It took 54 minutes.

Early stopping does not make the model better. It only stops the waste and keeps the right version. To get a lower val loss, the overfitting itself has to shrink, for example with more dropout.

## Dropout 0.3 on the GPU

Same model, same seed, same 5,000 steps, only dropout raised from 0.2 to 0.3.

| Dropout | Best val loss | At step | Train loss there | Gap there | Final gap |
| --- | --- | --- | --- | --- | --- |
| 0.2 | **1.4825** | 4,000 | 1.134 | 0.35 | 0.41 |
| 0.3 | 1.4863 | 4,000 | 1.187 | 0.30 | 0.35 |

- **The gap shrank** (0.41 to 0.35 at the end). More dropout did make the model memorize less.
- **The val loss did not improve.** 1.4863 against 1.4825 is a difference of 0.004, about the size of the wobble between estimates. The train loss went up by the same amount the gap went down.
- **Both runs hit their best at step 4,000** and then flattened.
- So dropout only moves the train loss. The val loss sits at about 1.48 either way. That points to the data as the limit: with 1 million characters, there is only so much this model can learn that carries over to unseen text.
- `train.py` goes back to dropout 0.2, the slightly better and simpler setting.

Text from the dropout 0.3 model:

```text
GLOUCESTER:
I would fought you not good Bohemia,
It will make me peril.

PRINCE EDWARD:
Peace, to my manster. For you believe
The mease to my chamberland with Buckingham;
```

## More training text (`data/prepare.py`)

The GPU model got stuck at a val loss of about 1.48 whatever the dropout, and the train loss kept falling: it was running out of new things to learn from 1 million characters. The fix to try is more text.

### Where the val data comes from

The val data is the last 10% of `input.txt`. Looking at the speakers in it (Petruchio, Katharina, Prospero, Miranda...), it is the end of **The Taming of the Shrew** and all of **The Tempest**. The train part is mostly Coriolanus, Richard II and III, Henry VI part 3, Measure for Measure, Romeo and Juliet and The Winter's Tale.

So the extra text must never contain those two plays. If it did, the model would train on the val text, and the val loss would look better without the model being better. That is called **leakage**.

### What `prepare.py` does

`python gpt/data/prepare.py` downloads Shakespeare's complete works from Project Gutenberg (5.6 MB) and writes `gpt/data/extra.txt`:

1. **Cuts the book into works** using the list of titles at the top.
2. **Leaves out** the poems (no speakers, so they do not look like a play), the two val plays, and every play that is already in `input.txt`.
3. **Cleans each play** into the format of `input.txt`:

| Gutenberg | `input.txt` style |
| --- | --- |
| `CLEOPATRA.` | `CLEOPATRA:` |
| `’` `‘` `“` `”` | `'` |
| `—` | `--` |
| ` Enter Antony.` and `[_Exeunt._]` | removed (stage directions) |
| `ACT I`, `SCENE II. Rome.` | removed |
| List of scenes and characters | removed |
| `é`, `æ` | `e`, `ae` |
| Any other character not in the 65 | removed |

The 65 characters stay the same, so the model and `generate.py` do not change.

### Finding the plays that are already in `input.txt`

`input.txt` comes from a different edition, so its lines are not word for word the same as Gutenberg's. Comparing whole lines found nothing. Instead, `prepare.py` cuts both texts into **runs of 6 words** (lower case, letters only) and checks what share of a play's runs also appear in `input.txt`:

| Play | Runs also in `input.txt` |
| --- | --- |
| The Winter's Tale | 89% |
| Romeo and Juliet | 85% |
| Measure for Measure | 83% |
| Richard II, Coriolanus, Henry VI part 3 | 80 to 81% |
| Richard III | 62% |
| Every other play | 0% |

The gap is so clear that any threshold between 1% and 60% would work; `prepare.py` uses 10%.

Checks on the result:

- 29 plays, **3,567,097 characters**. Together with the train part of `input.txt` that is 4,570,950 training characters, **4.5 times** as much as before.
- Only 0.08% of the val text's 6-word runs appear anywhere in `extra.txt`: common phrases, not copied text.
- No `[`, `]`, `_`, `Contents` or `Dramatis Personae` left.

### Using it

`train.py` has `use_extra = True`. The extra text is added to `train_data` only. `val_data` is exactly the same as before, so the val loss is directly comparable with the 1.48 of the earlier runs. Everything else on the GPU stays the same: same model, dropout 0.2, 5,000 steps.

With 4.5 times the text, 5,000 steps of 64 chunks of 256 characters go through the training text about 18 times instead of about 80. The model should run out of new things to learn much later, so the val loss should keep falling for longer.

### First run: the small model by accident

The first Kaggle run with the extra text had the GPU switched off, so `train.py` printed `device: cpu` and trained the small CPU model (161,985 parameters, 10,000 steps). Not the run that was planned, but a useful one:

| Small model | Best val loss | Train/val gap at the end |
| --- | --- | --- |
| Without extra text | 1.6600 | 0.20 |
| With extra text | 1.6566 | 0.11 |

- **The val loss hardly moved.** The small model was never short of text. It is limited by its size: 162 thousand numbers can only hold so much.
- **The gap halved** and the val loss set a new best at every estimate up to the end. With more text there is less to memorize.
- So more text helps a model that has run out of text, like the 10.8 million parameter GPU model, not one that has run out of room.

### Results: the GPU model with 4.5 times the text

Same model, same seed, dropout 0.2, 5,000 steps. Only the training text changed.

| Step | Train loss | Val loss, `input.txt` only | Val loss, with extra text |
| --- | --- | --- | --- |
| 1,000 | 1.54 / 1.65 | 1.73 | 1.74 |
| 2,000 | 1.32 / 1.42 | 1.56 | 1.53 |
| 3,000 | 1.21 / 1.32 | 1.50 | 1.45 |
| 4,000 | 1.13 / 1.27 | 1.48 | 1.40 |
| 5,000 | 1.07 / 1.24 | 1.48 | **1.37** |

(Train loss: without / with the extra text.)

| Model | Best val loss | Train/val gap at the end |
| --- | --- | --- |
| GPU model, `input.txt` only | 1.4825 | 0.41 |
| GPU model, dropout 0.3 | 1.4863 | 0.35 |
| **GPU model, with extra text** | **1.3666** | **0.13** |

- **The biggest single gain since residual connections.** 0.12 lower than anything before, from data alone, without touching the model.
- **The gap shrank from 0.41 to 0.13.** The train loss is higher (1.24 against 1.07) because there is far more text to remember, so the model has to learn patterns instead.
- **Up to step 1,500 the two runs are the same.** The extra text only starts to matter once the old run starts to run out of new things to learn.
- **The val loss was still falling at step 5,000** (1.3770 to 1.3666 in the last 500 steps), and every estimate was a new best. Early stopping never came close to stopping. Training longer should bring it lower.
- Compare with dropout 0.3: it shrank the gap by pushing the train loss up, and the val loss stayed the same. More text shrank the gap by pulling the val loss down. That is the difference between hiding memorization and giving the model something better to learn.

Text from the model trained with the extra text (it now knows characters from the new plays: Pucelle from Henry VI part 1, Beatrice from Much Ado About Nothing):

```text
PUCELLE:
An earth.

BOY:
But fast the comfort of that heavens with thy flood!

BEATRICE:
This miragies, if priest us, thy fault,
Can honour be seg of dissent, at the wind
```

### Results: 10,000 steps with the extra text

The same run, twice as long.

| Step | Train loss | Val loss |
| --- | --- | --- |
| 5,000 | 1.24 | 1.37 |
| 6,000 | 1.21 | 1.35 |
| 7,000 | 1.18 | 1.33 |
| 8,000 | 1.16 | 1.31 |
| 9,000 | 1.15 | 1.30 |
| 10,000 | 1.13 | **1.297** |

| GPU model | Steps | Best val loss | Gap at the end |
| --- | --- | --- | --- |
| `input.txt` only | 5,000 | 1.4825 | 0.41 |
| With extra text | 5,000 | 1.3666 | 0.13 |
| **With extra text** | **10,000** | **1.2970** | **0.17** |

- **Another 0.07 lower** from training twice as long. Altogether the extra text and the longer training took the val loss from 1.48 to 1.30.
- **It is slowing down.** The val loss fell 0.07 between steps 5,000 and 10,000, but only 0.006 in the last 500 steps. Step 8,500 was the first estimate without a new best (1.3175 against 1.3111), and the model recovered by step 9,000.
- **The gap is growing again** (0.13 at step 5,000, 0.17 at step 10,000). The model is starting to run out of new things in 4.5 million characters too, but much later than with 1 million.
- Early stopping never triggered: there was never more than one estimate in a row without a new best.

## Learning rate decay (`train.py`)

Until now the learning rate stayed the same for the whole run. Early on, big steps are useful because the model is far from good. Near the end the model is close, and big steps make it jump around the best point instead of settling into it. So the learning rate now starts high and shrinks.

```python
min_lr = learning_rate / 10

def get_lr(step):
    progress = step / max_steps   # 0 at the start, 1 at the end
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)
```

It follows half a cosine wave:

| Step (of 10,000) | `cos(pi * progress)` | GPU learning rate |
| --- | --- | --- |
| 0 | 1 | 3.0e-4 |
| 2,500 | 0.71 | 2.6e-4 |
| 5,000 | 0 | 1.65e-4 |
| 7,500 | -0.71 | 0.7e-4 |
| 10,000 | -1 | 0.3e-4 |

- **Why a cosine and not a straight line:** it stays high for longer at the start and flattens out at the end, so the model gets many big steps first and a long gentle landing.
- **`min_lr` is a tenth of the start**, so learning never stops completely.
- **Every step sets the rate** with `group['lr'] = get_lr(steps)`. The optimizer keeps its settings in `param_groups`, a list with one group here that holds every parameter.
- The rate is printed at every estimate, so the log shows it going down.
- `lr_decay = False` switches back to a fixed learning rate (and is the setting now, see the results below).
- The starting learning rate is unchanged (3e-4 on the GPU, 1e-3 on the CPU), so this run differs from the 1.2970 run in one thing only.

### Results: decay made it worse

| Step | Fixed learning rate | With decay | Learning rate with decay |
| --- | --- | --- | --- |
| 2,500 | 1.4883 | 1.4944 | 2.6e-4 |
| 5,000 | 1.3685 | 1.3772 | 1.65e-4 |
| 7,500 | 1.3247 | 1.3399 | 0.7e-4 |
| 9,500 | 1.3026 | **1.3221** (best) | 0.3e-4 |
| 10,000 | **1.2970** | 1.3282 | 0.3e-4 |

- **The decay run was behind at every estimate from step 3,000**, and the gap kept growing: 0.006 at step 2,500, 0.03 at the end.
- **Why:** decay pays off when a model has stopped improving at a fixed learning rate and is jumping around its best point. This model never got there. With a fixed rate it was still improving at step 10,000. Shrinking the steps only slowed it down: by step 5,000 the rate was already half, by step 7,500 under a quarter.
- **The train loss is higher too** (1.17 against 1.13), so the model really learned less. It is not a case of memorizing less.
- **Not kept:** `lr_decay = False`. The code stays, because decay would make sense for a run long enough to level off.
- Decay usually comes with a higher starting learning rate, so the model learns fast first and settles later. This run kept the start at 3e-4 to change one thing at a time. A higher start (6e-4 or 1e-3) with decay would be a separate experiment.

Text from the decay model:

```text
ALBANY:
How fares thee stands else?

MALCOLM:
Thou art villain, great Mars, what a fearful Duke.
```

### Next run: a higher start with decay

Decay usually comes with a higher starting learning rate. The GPU run now starts at **6e-4** (twice the old fixed rate) and decays to 6e-5:

| Step (of 10,000) | Learning rate |
| --- | --- |
| 0 | 6.0e-4 |
| 2,500 | 5.2e-4 |
| 5,000 | 3.3e-4 |
| 7,500 | 1.4e-4 |
| 10,000 | 0.6e-4 |

For the first half of the run the steps are bigger than the fixed 3e-4, so the model should learn faster early on, and the small steps only come at the end. The fixed-rate run gave 1.2970, and running it again gave 1.2975. (Two runs of the 6e-4 setting later differed by 0.005, so 0.0005 was luck: the real run-to-run noise is about 0.005, see below.)

### Results: a higher start with decay works

| Step | Fixed 3e-4 | Decay from 3e-4 | Decay from 6e-4 | Learning rate (6e-4 run) |
| --- | --- | --- | --- | --- |
| 1,000 | 1.7362 | 1.7367 | **1.6294** | 5.9e-4 |
| 2,500 | 1.4882 | 1.4944 | **1.4149** | 5.2e-4 |
| 5,000 | 1.3686 | 1.3772 | **1.3173** | 3.3e-4 |
| 7,500 | 1.3236 | 1.3399 | **1.2831** | 1.4e-4 |
| 9,500 | 1.2996 | 1.3221 | **1.2715** (best) | 0.6e-4 |
| 10,000 | 1.2975 | 1.3282 | 1.2794 | 0.6e-4 |

- **Best val loss 1.2715**, against 1.297 with the fixed rate. A second run with exactly the same settings gave **1.2766**, so identical runs differ by about 0.005 (the GPU adds numbers in a slightly different order each time, and the small differences grow over 10,000 steps). Over two runs each: fixed 3e-4 = 1.2970 / 1.2975, 6e-4 with decay = 1.2715 / 1.2766. The gain is 0.02 to 0.026, four to five times the noise, so it is real, but the honest way to state the result is **1.27 to 1.28**, not 1.2715.
- **Ahead at every estimate.** The bigger steps at the start did most of the work: by step 1,000 this run was already 0.11 ahead. It reached the old final result (1.297) around step 6,000, so the same result now takes 40% fewer steps.
- **The small steps at the end did their job too.** The val loss kept improving while the rate dropped below 1e-4, where the decay-from-3e-4 run had stalled.
- **The gap is a bit bigger** (train 1.09, val 1.27, gap 0.18 at the best step). Faster learning also means faster memorizing.
- **The last estimate went up again** (1.2715 to 1.2794). Early stopping keeps the step 9,500 model, so that does not matter.
- **Kept:** `train.py` uses a 6e-4 start with decay on the GPU.

Lesson from the two decay runs: decay is not a free improvement on its own. It pays off together with a higher start, because then the early steps are bigger than before and the late steps smaller.

## Measuring honestly: val, test and noise (`evaluate.py`)

Every choice so far (dropout, learning rate, how long to train, which checkpoint to keep) was made by looking at the val loss. So the best val number is a little optimistic: it is the best of several tries, all scored on the same text. Three fixes:

1. **Noise.** Two runs of exactly the same settings differ by about 0.005 (see the decay results). Differences smaller than that mean nothing.
2. **A test set that was never used for any choice.** Every play is already used for training or val. The only Shakespeare left are the **poems** (the Sonnets, Venus and Adonis, The Rape of Lucrece and three short ones), which `prepare.py` always skipped. It now writes them to `gpt/data/test.txt`: 265,764 characters, and only 0.11% of their 6-word runs appear anywhere in the training text. Poems are verse without speakers, a different style from the plays, so the test loss is not comparable with the val loss. It answers one question: does the model work on Shakespeare it never saw and nobody tuned on?
3. **The same number every time.** `estimate_loss()` in `train.py` uses 200 random batches, so it wobbles. `gpt/evaluate.py` goes through the whole val and test text in windows of `block_size` characters, one after the other, so the result is exact and repeatable.

`python gpt/evaluate.py` takes under 2 minutes on the laptop CPU.

| Model | Val (whole text) | Test (poems) |
| --- | --- | --- |
| Fixed 3e-4, 10,000 steps (`train.py` estimate: 1.2970) | 1.2958 | 1.3566 |

- The exact val loss (1.2958) is close to the estimate from random batches (1.2970).
- The poems score 0.06 worse than the val plays: a different style, but no collapse. The model learned Shakespeare's English, not just the play format.
- The test number is measured once per model and never used to choose anything. Otherwise it would become a second val set.

## KV cache (`model.py`, `benchmark.py`)

### The problem

`generate` used to run the whole context (up to 256 characters) through the model for every new character, then throw away every prediction except the last one. But the keys and values of the earlier characters never change: a character's key and value only depend on that character and the ones before it. So for each new character the model recomputed 255 sets of keys and values it already had.

The cost per character grows with the context length T. Attention compares every position with every other, so the work per new character is about T x T, and the whole text costs about T x T x (number of characters).

### The fix

Keep the keys and values: a **kv cache**, one `[k, v]` pair per head per block.

```python
if cache is not None:
    if cache[0] is not None:
        k = torch.cat([cache[0], k], dim=1)   # old keys + the new one
        v = torch.cat([cache[1], v], dim=1)
    cache[0], cache[1] = k, v
```

- **Prefill:** the prompt goes through once, in one pass, and fills the cache.
- **Every step after that:** only the newest character goes through the model. Its query looks at all the stored keys, so the work per character is about T instead of T x T.
- **Positions:** the newest character needs its real position, so `forward` takes `start_pos` and uses `torch.arange(start_pos, start_pos + T)` for the position embeddings.
- **The mask:** the T new queries are the last T of the Tk positions, so the mask is `tril[Tk - T:Tk, :Tk]`. Without a cache Tk == T and that is the old `tril[:T, :T]`. With one new character it is a single row of ones: it may look at everything before it.
- **When the context is full:** the position table only has 256 rows. Without a cache the context is cut to the last 256 characters every step, which shifts every position by one. A cache cannot shift the positions of what it already stored, so when it is full it starts again from the last 128 characters. Up to 256 characters the text is exactly the same as without a cache. After that it is different, but just as valid.
- `generate_stream` **yields** each character as soon as it is made (used by the server). `generate` is now a loop over it, so `train.py` and `generate.py` did not change.

### Is it still the same model?

`benchmark.py` checks two things before measuring anything:

- The logits for 256 positions, all at once without a cache against one at a time with a cache: largest difference **1.3e-05**, just the order floating point numbers are added in.
- With the same seed, the first 255 generated characters are **identical** with and without the cache.

### How much faster (laptop CPU, 4 threads, 10.8M parameter model, median of 3 runs)

| Method | 250 characters | 1,000 characters |
| --- | --- | --- |
| Before: no cache, no `no_grad` | 16 per second | |
| No cache, with `no_grad` | 16 per second | 10 per second |
| **KV cache** | **60 per second** | **58 per second** |

- **3.8 times faster for 250 characters, 5.5 times for 1,000.** The longer the text, the bigger the gain: without a cache it slows down as the context fills up (16 to 10 per second), with a cache the speed stays the same.
- **`no_grad` alone made no measurable difference.** Without it PyTorch keeps what a backward pass would need, but for one sequence at a time that is small. It is on anyway, because generating never needs gradients.
- 58 characters per second is still slow, because each step runs 36 small heads one after another in Python. On a CPU, at this size, the Python overhead is now a large share of the time.

## Batching the heads (tried, not kept)

The heads in `MultiHeadAttention` run one after another in a Python loop. The idea was that doing all of them at once would make every step faster.

The batched version glued the key, query and value weights of every head into one matrix, did one matrix multiply for all of them, then cut the result back into heads:

```python
W = torch.cat([h.key.weight for h in self.heads] + [h.query.weight ...] + [h.value.weight ...])
kqv = x @ W.T                                             # (B, T, 3 * H * head_size)
k, q, v = [part.view(B, T, H, hs).transpose(1, 2)         # (B, H, T, head_size)
           for part in kqv.split(H * hs, dim=-1)]
```

The rest is `Head.forward` with one more dimension in front for the heads. It gave exactly the same output as the loop on the CPU, and a difference of 0.0000017 on the GPU (the GPU adds numbers in a different order).

Time for one training step:

| Where | Model | Loop | Batched |
| --- | --- | --- | --- |
| Laptop CPU | Context 64, size 64 | 92 ms | 84 ms |
| Laptop CPU | Context 256, size 384 | 13.6 s | 14.9 s |
| Kaggle GPU | Context 256, size 384 | 453 and 517 ms | 434 and 503 ms |

- **About 3% faster on the GPU, and nothing on the CPU.** The loop over 6 heads was never the slow part. The time goes into the attention scores (every position against every other, 256 x 256 per head) and the feed-forward layers, and those are the same amount of work either way.
- **A first try was slower than the loop**: it multiplied `(B, 1, T, C)` by the stacked weights `(H, C, head_size)`, which PyTorch turns into many small matrix multiplies.
- **Not kept.** 3% is not worth the harder-to-read code, so `model.py` keeps the loop.
- **Measure before optimizing.** The plan assumed the loop was slow. Timing showed it was not.

## Mistakes I made and fixed

- **`train.py` read itself.** It opened `"gpt/train.py"` instead of `"gpt/data/input.txt"`. It was easy to miss because the script happened to contain exactly 65 unique characters too, so `vocab_size` printed 65 either way. The giveaway was the length: about 3,000 characters instead of 1,115,393.
- **`print(text)` on the full file** would print over a million characters. It is now `print(text[:1000])`.
- **`model.py` used names it did not have.** `vocab_size`, `xb`, `yb` and `decode` live in `train.py`, so those lines moved there.
- **Printing the loss on every step** flooded the terminal. It now prints only when `steps % 1000 == 0`.
- **Stray text typed into the code.** The word `do` ended up after the colon in `if steps % 1000 == 0:`, which gave `IndentationError: unexpected indent` on the next line. The error points at the line after the real mistake.

- **`data.type` without brackets** printed the method itself, not the type. It is now `data.dtype`.
- **The bigram line in the plot was a typed-in 2.50.** `train.py` now measures it: it counts how often each character follows each other character in the training text and scores the val text with those odds (`bigram_val_loss()`). That is the best any bigram can do on this data: 2.48 on `input.txt` alone, matching the trained bigram, and 2.46 with the extra text.

## Still to fix

- **`generate.py` only works with the full model now.** `train.py` saves the settings (`n_embd`, `block_size`, `n_head`, `n_layer`, `dropout`) inside `gpt/gpt.pt`, and `generate.py` rebuilds the model from them. To generate from the old bigram instead, import `BigramLanguageModel` from `bigram` and load `gpt/bigram.pt`.

## Next steps

Done: the bigram baseline, attention, all five transformer pieces, scaling up on the CPU, the tutorial-size model on a GPU (val loss 1.48), early stopping, 4.5 times more training text, 10,000 steps on the GPU (val loss 1.30), and a higher learning rate that decays (val loss 1.27).

1. The GPU model is at 1.2715. The val loss was still improving at the end, so a longer run with the same schedule (for example 15,000 steps) is the cheapest next try.
2. More text again: all of Shakespeare's plays are already used, so it would have to be other authors.
3. Try dropout on the CPU model. Its train/val gap reached 0.20 after 10,000 steps, so it may start to pay off now.
