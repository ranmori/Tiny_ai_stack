

import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
#  create a lower triangular matrix
tril = torch.tril(torch.ones(T, T))              # lower triangle of ones
# create a weight matrix for attention
wei = torch.zeros(T, T)
# compute the attention weights
wei = wei.masked_fill(tril == 0, float('-inf'))  # block the future
# apply softmax to get probabilities
wei = F.softmax(wei, dim=-1)                     # each row becomes an even average of the past
out = wei @ x                                    # (T, T) @ (B, T, C) -> (B, T, C)

print('----- step 1: the even average -----')
print(wei)
# keep a copy for the plot, step 2 reuses the name wei
wei_average = wei


# step 2: a single self attention head
# in step 1 wei started as zeros, so every earlier character got the same weight
# now each character decides for itself how much to listen to each earlier character
import torch.nn as nn

torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch, time, channels: every character is now described by 32 numbers
x = torch.randn(B, T, C)

# how many numbers the query, key and value vectors have
head_size = 16
# three small linear layers, each turns the 32 numbers of a character into 16
# no bias, it is only a matrix multiply
key = nn.Linear(C, head_size, bias=False)    # what do i contain?
query = nn.Linear(C, head_size, bias=False)  # what am i looking for?
value = nn.Linear(C, head_size, bias=False)  # what i hand over if someone listens to me

k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)

# the score between two characters is the query of one times the key of the other
# a big score means "what i am looking for matches what you contain"
# transpose(-2, -1) swaps the last two dimensions of k so the shapes line up:
# (B, T, 16) @ (B, 16, T) -> (B, T, T), one score for every pair of positions
# the scores are divided by the square root of head_size to keep them small,
# big scores would make softmax put almost everything on a single character
wei = q @ k.transpose(-2, -1) * head_size**-0.5

# the same two lines as step 1: block the future, then turn each row into weights that add up to 1
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# each character collects the values of the characters it listens to, not the raw x
v = value(x)   # (B, T, 16)
out = wei @ v  # (B, T, T) @ (B, T, 16) -> (B, T, 16)

print('----- step 2: self attention -----')
print(wei[0]) # the weights of the first chunk in the batch, no longer an even average
print(out.shape)


# plot the weights before attention (the even average) next to the weights with attention
# detach because wei came out of the linear layers and still carries gradient information
from plots import plot_attention
plot_attention(wei_average, wei[0].detach())
