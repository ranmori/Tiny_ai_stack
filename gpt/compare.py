
# what does each piece of the transformer add?
# train the model six times, switching on one more piece each time, and compare the losses
# run from the project root: python gpt/compare.py  (it trains six models so it takes a while)
import torch
from model import GPTLanguageModel
from plots import plot_comparison

# the same data steps as train.py
with open("gpt/data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()
char = sorted(list(set(text)))
vocab_size = len(char)
stoi = {ch:i for i, ch in enumerate(char)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# the same settings as train.py
batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
# fewer steps than train.py because there are six models to train
max_steps = 5000
eval_interval = 500
eval_iters = 100

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(m):
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

# each row switches on one more piece than the row above it
variants = [
    ('4 heads only',        dict(n_layer=1, dropout=0.0, feed_forward=False, residual=False, layer_norm=False)),
    ('+ feed forward',      dict(n_layer=1, dropout=0.0, feed_forward=True,  residual=False, layer_norm=False)),
    ('+ 3 stacked blocks',  dict(n_layer=3, dropout=0.0, feed_forward=True,  residual=False, layer_norm=False)),
    ('+ residual',          dict(n_layer=3, dropout=0.0, feed_forward=True,  residual=True,  layer_norm=False)),
    ('+ layer norm',        dict(n_layer=3, dropout=0.0, feed_forward=True,  residual=True,  layer_norm=True)),
    ('+ dropout 0.2',       dict(n_layer=3, dropout=0.2, feed_forward=True,  residual=True,  layer_norm=True)),
]

results = []
for name, settings in variants:
    # the same seed for every model so they start from the same random numbers and see the same batches
    torch.manual_seed(1337)
    # ** unpacks the dict into named arguments: n_layer=1, dropout=0.0, ...
    m = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, **settings)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # numel is the number of elements in a tensor, summed over every parameter it is the size of the model
    n_params = sum(p.numel() for p in m.parameters())
    print(f'----- {name} ({n_params} parameters) -----')

    steps, train_losses, val_losses = [], [], []
    for step in range(max_steps + 1):
        if step % eval_interval == 0:
            est = estimate_loss(m)
            steps.append(step)
            train_losses.append(est['train'])
            val_losses.append(est['val'])
            print(f"step {step}: train loss {est['train']:.4f}, val loss {est['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    results.append({'name': name, 'params': n_params, 'steps': steps, 'train': train_losses, 'val': val_losses})

# the final numbers side by side
print('----- final -----')
for r in results:
    print(f"{r['name']:20s} {r['params']:6d} parameters   train {r['train'][-1]:.4f}   val {r['val'][-1]:.4f}")

plot_comparison(results)
