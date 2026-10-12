# import torch
# from model import BigramLanguageModel

# #load what train.py saved 

# checkpoint= torch.load("gpt/bigram.pt")
# char= checkpoint['char']
# itos={i:ch for i,ch in enumerate(char)}
# decode= lambda l: ''.join([itos[i] for i in l ])


# #rebuild the model with the same vocab size and load the state dict
# m= BigramLanguageModel(vocab_size= len(char))
# m.load_state_dict(checkpoint['model_state_dict'])
# m.eval()


# contex= torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(idx= contex, max_new_tokens=500)[0].tolist()))


