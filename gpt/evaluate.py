# measure the saved model on the val text and on the held-out test text
# run from the project root: python gpt/evaluate.py
#
# val:  the last 10% of input.txt (end of The Taming of the Shrew, The Tempest)
#       every choice so far (learning rate, dropout, which checkpoint to keep) was made by looking at it,
#       so the val loss is a little optimistic
# test: the poems (gpt/data/test.txt, made by gpt/data/prepare.py), never trained on and never used for any choice
#       verse without speakers, so a different style from the plays: compare it with nothing but itself
#
# unlike estimate_loss() in train.py, which uses random batches, this goes through the whole text
# in windows of block_size characters, one after the other, so the number is exactly the same every run
import torch
from checkpoint import load_checkpoint

m, char, encode, decode = load_checkpoint()
block_size = m.block_size

with open('gpt/data/input.txt', encoding='utf-8') as f:
    text = f.read()
val_text = text[int(0.9 * len(text)):]
with open('gpt/data/test.txt', encoding='utf-8') as f:
    test_text = f.read()


@torch.no_grad()
def loss_on(text, batch_size=16):
    data = torch.tensor(encode(text), dtype=torch.long)
    # windows that do not overlap: inputs data[i : i + block_size], targets one character later
    starts = list(range(0, len(data) - block_size - 1, block_size))
    total, count = 0.0, 0
    for b in range(0, len(starts), batch_size):
        ix = starts[b:b + batch_size]
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        _, loss = m(x, y)
        # loss is the mean over this batch, weight it by how many characters it covers
        total += loss.item() * y.numel()
        count += y.numel()
    return total / count


print(f'val loss:  {loss_on(val_text):.4f}   ({len(val_text):,} characters, the end of The Shrew and The Tempest)')
print(f'test loss: {loss_on(test_text):.4f}   ({len(test_text):,} characters, the poems, never trained on)')
