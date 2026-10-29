import torch
from checkpoint import load_checkpoint

# load what train.py saved, rebuilt into a model in eval mode (dropout off)
# if gpt/gpt.pt is missing it is downloaded from the GitHub release first, see checkpoint.py
m, char, encode, decode = load_checkpoint()

# start from one newline character (index 0)
contex= torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx= contex, max_new_tokens=500)[0].tolist()))
