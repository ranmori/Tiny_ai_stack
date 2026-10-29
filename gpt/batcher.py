# request batching for server.py: one worker thread runs the requests of several users through the model together
#
# without batching every request runs on its own thread, with its own forward pass of batch size 1
# on a CPU with 4 cores they get in each other's way: with 8 users at once the server made fewer characters
# per second than with 2
# with batching the requests wait in a queue, the worker takes everything that arrived within a few milliseconds
# (up to max_batch) and makes the next character for all of them in one forward pass of batch size B
#
# the prompts have different lengths, but a batch is one tensor, so the shorter ones are padded on the left:
#   row 0:  [pad pad R O M E O :]
#   row 1:  [J U L I E T : \n]
# pad_mask marks the padding, so no real character ever listens to it,
# and pos gives every row its own positions, so the first real character of every row is at position 0
#
# this is static batching: a batch runs until all its requests are done, new requests wait for the next batch
# a request that finishes leaves the batch (its rows are removed from the cache), so the others get faster
import queue
import threading
import time
import torch
from torch.nn import functional as F


class Request:
    def __init__(self, tokens, max_new_tokens, generator):
        self.tokens = list(tokens)          # the prompt, and every new character is added to it
        self.remaining = max_new_tokens
        self.generator = generator          # every request keeps its own random numbers
        self.out = queue.Queue()            # new characters go here, then None when it is done


class Batcher:

    def __init__(self, model, max_batch=8, wait=0.01):
        self.m = model
        self.max_batch = max_batch
        self.wait = wait
        self.waiting = queue.Queue()
        self.batch_sizes = []  # how big every batch was, for /health
        threading.Thread(target=self.worker, daemon=True).start()

    def submit(self, tokens, max_new_tokens, generator):
        request = Request(tokens, max_new_tokens, generator)
        self.waiting.put(request)
        return request

    def worker(self):
        while True:
            # wait for the first request, then collect whatever else arrives within self.wait seconds
            batch = [self.waiting.get()]
            deadline = time.perf_counter() + self.wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.waiting.get(timeout=max(0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            self.batch_sizes.append(len(batch))
            try:
                self.run(batch)
            except Exception as e:
                # never leave a user waiting forever
                for r in batch:
                    r.out.put(e)

    def prefill(self, contexts):
        # pad the contexts on the left to the same length, then run them through the model and fill a new cache
        lengths = torch.tensor([len(c) for c in contexts])
        L = int(lengths.max())
        pads = L - lengths
        idx = torch.zeros((len(contexts), L), dtype=torch.long)
        for i, c in enumerate(contexts):
            idx[i, L - len(c):] = torch.tensor(c)
        pad_mask = torch.arange(L)[None, :] < pads[:, None]          # (B, L), True on the padding
        pos = (torch.arange(L)[None, :] - pads[:, None]).clamp(min=0)  # (B, L), every row starts at position 0
        cache = self.m.new_cache()
        logits, _ = self.m(idx, cache=cache, pos=pos, pad_mask=pad_mask)
        return logits[:, -1, :], cache, pad_mask, lengths

    @torch.no_grad()
    def run(self, batch):
        block_size = self.m.block_size
        rows = batch
        logits, cache, pad_mask, next_pos = self.prefill([r.tokens[-block_size:] for r in rows])
        while True:
            # one new character for every row, each sampled with its own request's random numbers
            probs = F.softmax(logits, dim=-1)
            for i, r in enumerate(rows):
                token = torch.multinomial(probs[i:i + 1], 1, generator=r.generator).item()
                r.tokens.append(token)
                r.remaining -= 1
                r.out.put(token)
                if r.remaining == 0:
                    r.out.put(None)
            # finished requests leave the batch: keep only their rows of the cache
            keep = [i for i, r in enumerate(rows) if r.remaining > 0]
            if not keep:
                return
            if len(keep) < len(rows):
                rows = [rows[i] for i in keep]
                cache = [[[k[keep], v[keep]] for k, v in block] for block in cache]
                pad_mask, next_pos = pad_mask[keep], next_pos[keep]
            if pad_mask.shape[1] == block_size:
                # the cache is full: start again from the last block_size // 2 characters of every row
                logits, cache, pad_mask, next_pos = self.prefill([r.tokens[-(block_size // 2):] for r in rows])
                continue
            # only the newest character of every row, at that row's next position
            x = torch.tensor([[r.tokens[-1]] for r in rows])
            pad_mask = torch.cat([pad_mask, torch.zeros((len(rows), 1), dtype=torch.bool)], dim=1)
            logits, _ = self.m(x, cache=cache, pos=next_pos[:, None], pad_mask=pad_mask)
            logits = logits[:, -1, :]
            next_pos = next_pos + 1
