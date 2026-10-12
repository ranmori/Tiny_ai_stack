# GPT notes

Notes on what I have built so far in the `gpt` folder, and what comes next.

| File | What it is | State |
| --- | --- | --- |
| `data/input.txt` | The training text: about 1.1 million characters of Shakespeare | Ready |
| `train.py` | Loads the text, tokenizes it, makes batches, trains the attention model, estimates the loss, writes text, saves the model, calls the plots | Runs end to end |
| `model.py` | `Head`, `MultiHeadAttention` and `AttentionLanguageModel` | 4 heads working |
| `bigram.py` | `BigramLanguageModel`, the baseline to beat | Done, kept for reference |
| `plots.py` | One plot function per stage, plus two shared helpers | Done |
| `attention.py` | Scratch file for learning attention, no model in it | Steps 1 and 2 done |
| `generate.py` | Will load saved weights and write text without retraining | Commented out and out of date |
| `attention.pt`, `bigram.pt` | Trained models saved by `train.py` | Not on GitHub, `*.pt` is in `.gitignore` |

Run from the project root: `python gpt/train.py`. The plot window opens when training finishes. The scratch file runs with `python gpt/attention.py`.

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

`plots.py` now has three plot functions and two helpers they share:

| Function | Used by | What it draws |
| --- | --- | --- |
| `plot_training` | The bigram stage | Loss, and the lookup table before and after |
| `plot_attention` | `attention.py` | The even average next to one attention grid |
| `plot_attention_training` | `train.py` now | Loss, and every head's weights before and after |
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

`AttentionLanguageModel` is the bigram with four changes:

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

## Mistakes I made and fixed

- **`train.py` read itself.** It opened `"gpt/train.py"` instead of `"gpt/data/input.txt"`. It was easy to miss because the script happened to contain exactly 65 unique characters too, so `vocab_size` printed 65 either way. The giveaway was the length: about 3,000 characters instead of 1,115,393.
- **`print(text)` on the full file** would print over a million characters. It is now `print(text[:1000])`.
- **`model.py` used names it did not have.** `vocab_size`, `xb`, `yb` and `decode` live in `train.py`, so those lines moved there.
- **Printing the loss on every step** flooded the terminal. It now prints only when `steps % 1000 == 0`.
- **Stray text typed into the code.** The word `do` ended up after the colon in `if steps % 1000 == 0:`, which gave `IndentationError: unexpected indent` on the next line. The error points at the line after the real mistake.

## Still to fix

- **`data.type` without brackets** prints the method itself, not the type. Use `data.dtype`.
- **`generate.py` is commented out and out of date.** It imports `BigramLanguageModel` from `model`, which no longer has that class. For the bigram, import it from `bigram` and keep loading `gpt/bigram.pt`. For the attention model it needs `AttentionLanguageModel`, the same `n_embd`, `block_size` and `n_head` as training, and `gpt/attention.pt`.
- **The bigram line in the plot is a fixed number.** `baseline=2.50` is typed in at the bottom of `train.py`, not measured each run.
- **A stale comment in `estimate_loss()`** still says "none in the bigram yet".

## Next steps

Done: the bigram baseline, attention steps 1 to 4 (the scratch file, one head in the model, four heads), and the attention plot.

1. Add a feed-forward layer after the heads: a small MLP applied to every position by itself. Attention lets positions gather information; this gives each one a chance to think about what it gathered.
2. Wrap attention and feed-forward into a `Block`, and stack several blocks.
3. Add residual connections and layer norm, which make a deeper model trainable.
4. Add dropout to fight overfitting.
5. Scale up: larger `block_size`, `n_embd` and more layers. The full small transformer should reach a validation loss of roughly 1.5.
