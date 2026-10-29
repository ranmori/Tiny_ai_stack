# load the trained model saved by train.py, used by generate.py, evaluate.py, benchmark.py and server.py
# gpt.pt is not in git (too big), so if it is missing it is downloaded from the GitHub release first
import os
import urllib.request
import torch
from model import GPTLanguageModel

RELEASE_URL = 'https://github.com/ranmori/Tiny_ai_stack/releases/download/v1.0/gpt.pt'


def download(path, url=RELEASE_URL):
    print(f'{path} not found, downloading the trained model from {url}')
    def progress(blocks, block_size, total):
        done = min(blocks * block_size, total)
        print(f'\r  {done / 1e6:5.1f} of {total / 1e6:.1f} MB', end='', flush=True)
    urllib.request.urlretrieve(url, path + '.part', reporthook=progress)
    # only rename when complete, so a broken download never looks like a model
    os.replace(path + '.part', path)
    print()


def load_checkpoint(path='gpt/gpt.pt'):
    # gives back the model in eval mode (dropout off), plus encode and decode for its 65 characters
    if not os.path.exists(path):
        download(path)
    # map_location='cpu' so a model trained on a GPU also loads on a computer without one
    checkpoint = torch.load(path, map_location='cpu')
    char = checkpoint['char']
    stoi = {ch: i for i, ch in enumerate(char)}
    itos = {i: ch for i, ch in enumerate(char)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)
    # rebuild a model of exactly the same shape, then load the trained numbers into it
    # ** unpacks the settings dict into named arguments: n_embd=384, block_size=256, ...
    m = GPTLanguageModel(vocab_size=len(char), **checkpoint['settings'])
    m.load_state_dict(checkpoint['model_state_dict'])
    m.eval()
    return m, char, encode, decode
