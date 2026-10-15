import torch
from model import GPTLanguageModel

#load what train.py saved

checkpoint= torch.load("gpt/gpt.pt")
char= checkpoint['char']
itos={i:ch for i,ch in enumerate(char)}
decode= lambda l: ''.join([itos[i] for i in l ])


#rebuild the model with the same vocab size and the same settings as in training, then load the state dict
# ** unpacks the settings dict into named arguments: n_embd=32, block_size=8, ...
m= GPTLanguageModel(vocab_size= len(char), **checkpoint['settings'])
m.load_state_dict(checkpoint['model_state_dict'])
# eval mode switches dropout off, so the whole model is used for writing
m.eval()


contex= torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx= contex, max_new_tokens=500)[0].tolist()))


