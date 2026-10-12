

import torch # the entire library tensors and math
# the linear and embeddings
import torch.nn as nn
# soft max and cross entropy
from torch.nn import functional as F

# for consistency
torch.manual_seed(1337)


    
# PyTorch's contract for "something with trainable parameters." It gives you: calling the object like a function, collecting parameters for the optimizer, moving to GPU, etc.
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        # intiliaze pytorch's internal machinery the book_keeping layers always required
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # so this the entire model or the look up table you create a random matrix shape a 65 by 65
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # Feed in character 7, read off row 7, 
    # and those 65 numbers are your prediction for the next character.
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # array lookup for each integer for each index take that row 
        logits = self.token_embedding_table(idx) # (B,T,C) [4,8,65]

        if targets is None:
            loss= None
        else:

            # Why? Because of what the loss function expects. Every position is a multiple-choice question: "given this context, which of 65 characters comes next?" 
            # 32 questions × 65 options, with 32 answers. cross_entropy wants exactly that shape: (questions, options) and (answers,).
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            # Alwaysreturns both. When grading: (32, 65) logits + a number. When generating: (B, T, 65) logits + None.

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens): # _ i dont need the count and new characters
            # get the predictions
            # self(idx) calls forward(idx) (that's the nn.Module magic).
            # No targets passed, so loss comes back None. 
            # logits has shape (B, T, C) — a prediction for every position in the sequence so far, not just the last one
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

