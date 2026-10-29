

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

    def forward(self, x, cache=None, pad_mask=None):
        # x is (B, T, n_embd)
        # cache (only while generating): a list [k, v] with the keys and values of every character before x
        # pad_mask (only for batches of prompts with different lengths, see batcher.py): (B, Tk), True where the
        # position is padding, so no real character listens to it
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        # kv cache: the keys and values of earlier characters never change, so keep them instead of recomputing them
        # the new characters' k and v go on the end, and only the new characters need a query
        if cache is not None:
            if cache[0] is not None:
                k = torch.cat([cache[0], k], dim=1) # (B, Tk, head_size), Tk = earlier characters + T
                v = torch.cat([cache[1], v], dim=1)
            cache[0], cache[1] = k, v
        Tk = k.shape[1]
        # the score between every pair of positions, divided by the square root of head_size to keep it small
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, Tk) -> (B, T, Tk)
        # block the future: the T queries are the last T of the Tk positions, so take those rows of tril
        # without a cache Tk == T and this is tril[:T, :T], cut down because the context can be shorter than block_size
        wei = wei.masked_fill(self.tril[Tk - T:Tk, :Tk] == 0, float('-inf'))
        if pad_mask is not None:
            # real characters never listen to padding
            # padding itself may listen to the padding before it, otherwise its whole row would be -inf and softmax gives NaN
            query_is_pad = pad_mask[:, Tk - T:] # (B, T)
            wei = wei.masked_fill(pad_mask[:, None, :] & ~query_is_pad[:, :, None], float('-inf'))
        # each row becomes weights that add up to 1
        wei = F.softmax(wei, dim=-1)
        # keep a copy of the weights so they can be plotted, detach so it carries no gradient information
        self.wei = wei.detach()
        # randomly stop some positions from talking to each other, only while training
        wei = self.dropout(wei)
        # each position collects the values of the positions it listens to
        out = wei @ v     # (B, T, Tk) @ (B, Tk, head_size) -> (B, T, head_size)
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

    def forward(self, x, cache=None, pad_mask=None):
        # every head gets the same x and gives back (B, T, head_size)
        # cat glues the results together along the last dimension:
        # 4 heads of 8 numbers each -> 32 numbers per position, the same size as n_embd
        # cache: one [k, v] per head, or None
        out = torch.cat([h(x, None if cache is None else cache[i], pad_mask) for i, h in enumerate(self.heads)], dim=-1)
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

    def forward(self, x, cache=None, pad_mask=None):
        # talk: normalize, then attention
        out = self.sa(self.ln1(x), cache, pad_mask)
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

    def forward(self, idx, targets=None, cache=None, start_pos=0, pos=None, pad_mask=None):

        # idx and targets are both (B,T) tensor of integers
        # cache and start_pos are only used while generating with a kv cache (see generate_stream):
        # cache holds one [k, v] per head per block, start_pos is the position of the first character in idx
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd) what each character is
        # pos and pad_mask are only for batches of prompts with different lengths (batcher.py):
        # pos (B,T) gives every row its own positions, pad_mask (B, Tk) marks the padding
        if pos is None:
            pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        pos_emb = self.position_embedding_table(pos) # (T,n_embd) or (B,T,n_embd), where each character sits
        # add them, pos_emb is broadcast across the batch so every chunk gets the same position vectors
        x = tok_emb + pos_emb # (B,T,n_embd)
        # through every block in turn: gather from the positions before, then think
        if cache is None:
            x = self.blocks(x) # (B,T,n_embd)
        else:
            for block, block_cache in zip(self.blocks, cache):
                x = block(x, block_cache, pad_mask)
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

    def new_cache(self):
        # an empty kv cache: for every block, for every head, a [keys, values] pair that starts empty
        return [[[None, None] for _ in block.sa.heads] for block in self.blocks]

    # no_grad: generating is not training, nothing needs to be remembered for backward
    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, use_cache=True, generator=None):
        # like generate, but hands out every new character as soon as it is made (yield), for streaming
        # idx is (B, T) array of indices in the current context
        # generator: an optional torch.Generator for the random sampling, so a server can give every request its own seed
        #
        # without a cache every new character runs the whole context through the model again,
        # so the work per character grows with the length of the context
        # with a kv cache only the newest character goes through the model, its query looks at the stored keys and values
        #
        # the position table only has block_size rows, so the context can only be block_size characters long
        # without a cache the context is cut to the last block_size characters every step, so every position shifts by one
        # a cache can not shift the positions of what it stored, so when it is full it starts again from the last
        # block_size // 2 characters: the text is exactly the same as without a cache up to block_size characters,
        # and different (but just as valid) after that
        cache = None
        refill = False
        pos = 0
        for _ in range(max_new_tokens):
            if not use_cache:
                # the old way: the whole context through the model, every time
                logits, _ = self(idx[:, -self.block_size:])
            elif cache is None:
                # the first step, or the cache just ran full: fill a new cache with the context (the prefill)
                context = idx[:, -(self.block_size // 2):] if refill else idx[:, -self.block_size:]
                cache = self.new_cache()
                logits, _ = self(context, cache=cache, start_pos=0)
                pos = context.shape[1]
            else:
                # only the newest character, at the next position, the rest comes from the cache
                logits, _ = self(idx[:, -1:], cache=cache, start_pos=pos)
                pos += 1
            # focus only on the last time step, turn it into probabilities and sample from them
            probs = F.softmax(logits[:, -1, :], dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            yield idx_next
            # the next character would need position block_size, which does not exist: start a new cache
            if use_cache and pos == self.block_size:
                cache = None
                refill = True

    def generate(self, idx, max_new_tokens, use_cache=True, generator=None):
        # idx is (B, T) array of indices in the current context
        # gives back idx with max_new_tokens new characters added, see generate_stream for how they are made
        for idx_next in self.generate_stream(idx, max_new_tokens, use_cache, generator):
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

