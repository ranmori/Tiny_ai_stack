

import torch # the entire library tensors and math
# the linear and embeddings
import torch.nn as nn
# soft max and cross entropy
from torch.nn import functional as F

# for consistency
torch.manual_seed(1337)


# one head of self attention, the same math as step 2 in attention.py but as a module so it can be trained
class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        # three small linear layers, no bias, it is only a matrix multiply
        self.key = nn.Linear(n_embd, head_size, bias=False)    # what do i contain?
        self.query = nn.Linear(n_embd, head_size, bias=False)  # what am i looking for?
        self.value = nn.Linear(n_embd, head_size, bias=False)  # what i hand over if someone listens to me
        # the lower triangle of ones that blocks the future
        # register_buffer: it belongs to the module but it is not a trainable parameter, the optimizer never touches it
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # dropout: while training, randomly switch off some of the attention weights (see the note above Block)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, T, n_embd)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # the score between every pair of positions, divided by the square root of head_size to keep it small
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # block the future, tril is cut down to T by T because while generating the context can be shorter than block_size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # each row becomes weights that add up to 1
        wei = F.softmax(wei, dim=-1)
        # keep a copy of the weights so they can be plotted, detach so it carries no gradient information
        self.wei = wei.detach()
        # randomly stop some positions from talking to each other, only while training
        wei = self.dropout(wei)
        # each position collects the values of the positions it listens to
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


# several heads of self attention side by side
# one head can only pay attention in one way, several heads can each look for something different
# (one might watch the previous character, another the last space, another the vowels...)
class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        # ModuleList is a python list that pytorch can see inside, so the parameters of every head get trained
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        # proj (projection): one linear layer that mixes what the separate heads found into a single answer
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # every head gets the same x and gives back (B, T, head_size)
        # cat glues the results together along the last dimension:
        # 4 heads of 8 numbers each -> 32 numbers per position, the same size as n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# piece 1: the feed forward layer
# attention is the positions talking to each other and gathering information
# feed forward is each position thinking on its own about what it gathered
# it is a small MLP like the one in the autograd folder, applied to every position separately
class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        # Sequential runs the layers one after the other
        self.net = nn.Sequential(
            # grow to 4 times the size to give it room to think
            nn.Linear(n_embd, 4 * n_embd),
            # the activation, without it the two linear layers would collapse into one
            nn.ReLU(),
            # shrink back to n_embd so the output has the same size as the input
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# piece 2: the block, one round of "talk to each other (attention), then think (feed forward)"
# blocks are stacked, so later blocks work with what earlier blocks figured out
#
# piece 3: the residual connections, the "x = x + ..." lines
# the block does not replace x, it adds a correction on top of it
# + passes the gradient straight through (like in the autograd engine), so even the first block
# gets a clear signal during backward, without it a stack of blocks trains badly
#
# piece 4: layer norm
# rescales the n_embd numbers of every position to mean 0 and spread 1 before attention and before feed forward
# the numbers stay in a steady range however deep the stack is, which keeps training stable
#
# piece 5: dropout
# while training, randomly set some numbers to zero, a different set on every step
# the model can not rely on any single path so it memorizes less (less overfitting)
# m.eval() switches it off for measuring and generating, m.train() switches it back on
#
# feed_forward, residual and layer_norm can be switched off, compare.py uses that to measure what each piece adds
class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout, feed_forward=True, residual=True, layer_norm=True):
        super().__init__()
        self.residual = residual
        self.sa = MultiHeadAttention(n_embd, n_head, n_embd // n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout) if feed_forward else None
        # Identity is a layer that does nothing, it stands in when layer norm is switched off
        self.ln1 = nn.LayerNorm(n_embd) if layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if layer_norm else nn.Identity()

    def forward(self, x):
        # talk: normalize, then attention
        out = self.sa(self.ln1(x))
        # residual: add the result on top of x instead of replacing x
        x = x + out if self.residual else out
        # think: normalize, then feed forward
        if self.ffwd is not None:
            out = self.ffwd(self.ln2(x))
            x = x + out if self.residual else out
        return x


# the full model: embeddings, a stack of blocks, a final layer norm, and the head that gives the logits
# this is a small GPT, the bigram is kept in bigram.py as the baseline to beat
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, feed_forward=True, residual=True, layer_norm=True):
        # intiliaze pytorch's internal machinery the book_keeping layers always required
        super().__init__()
        self.block_size = block_size
        # each token now reads off a vector of n_embd numbers that describes it, not the logits directly
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each position 0..block_size-1 also gets its own vector
        # attention only mixes vectors together, it has no idea of order, this is what tells it where a character sits
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # n_layer blocks one after the other
        # every block has n_head heads of self attention, each one n_embd // n_head numbers wide (4 heads of 8 for n_embd 32)
        # the * unpacks the list, Sequential wants the blocks as separate arguments
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, feed_forward, residual, layer_norm) for _ in range(n_layer)])
        # one last layer norm after the final block
        self.ln_f = nn.LayerNorm(n_embd) if layer_norm else nn.Identity()
        # lm_head (language model head) turns the n_embd numbers back into one score per character
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd) what each character is
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,n_embd) where each character sits
        # add them, pos_emb is broadcast across the batch so every chunk gets the same position vectors
        x = tok_emb + pos_emb # (B,T,n_embd)
        # through every block in turn: gather from the positions before, then think
        x = self.blocks(x) # (B,T,n_embd)
        x = self.ln_f(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size) [4,8,65]

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
            # the position table only has block_size rows, so only the last block_size characters can go in
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            # self(idx_cond) calls forward(idx_cond) (that's the nn.Module magic).
            # No targets passed, so loss comes back None.
            # logits has shape (B, T, C) — a prediction for every position in the sequence so far, not just the last one
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

