# how much faster is generating with a kv cache, and is the text still the same?
# run from the project root: python gpt/benchmark.py
import statistics
import time
import torch
from torch.nn import functional as F
from checkpoint import load_checkpoint

m, char, encode, decode = load_checkpoint()
block_size = m.block_size
print(f'model: {sum(p.numel() for p in m.parameters()):,} parameters, block_size {block_size}, '
      f'{torch.get_num_threads()} CPU threads')

# ----- 1. the cache gives the same answers -----

# the logits for every position: once with the whole sequence at once (no cache),
# once feeding one character at a time into a cache
torch.manual_seed(0)
seq = torch.randint(len(char), (1, block_size))
with torch.no_grad():
    full, _ = m(seq)
    cache = m.new_cache()
    step = [m(seq[:, t:t + 1], cache=cache, start_pos=t)[0] for t in range(block_size)]
diff = (full - torch.cat(step, dim=1)).abs().max().item()
print(f'largest difference in logits, cache against no cache: {diff:.2e}')
assert diff < 1e-4

# the same seed must give the same text, as long as it stays within block_size characters
start = torch.zeros((1, 1), dtype=torch.long)
n = block_size - 1
torch.manual_seed(1)
with_cache = decode(m.generate(start, n, use_cache=True)[0].tolist())
torch.manual_seed(1)
without_cache = decode(m.generate(start, n, use_cache=False)[0].tolist())
print(f'same {n} characters with and without the cache: {with_cache == without_cache}')
assert with_cache == without_cache


# ----- 2. speed -----

def generate_before(idx, max_new_tokens):
    # generate exactly as it was before the cache: no cache, and no torch.no_grad(),
    # so PyTorch also kept everything it would need for a backward pass that never comes
    for _ in range(max_new_tokens):
        logits, _ = m(idx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
    return idx


def tokens_per_second(fn, n_tokens, runs=3):
    fn(start, 5) # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(start, n_tokens)
        times.append(time.perf_counter() - t0)
    # the median of the runs, so one slow run does not decide the number
    return n_tokens / statistics.median(times)


rows = [
    ('before: no cache, no no_grad', generate_before, [250]),
    ('no cache (no_grad)', lambda idx, n: m.generate(idx, n, use_cache=False), [250, 1000]),
    ('kv cache', lambda idx, n: m.generate(idx, n, use_cache=True), [250, 1000]),
]
results = {}
for name, fn, lengths in rows:
    for n_tokens in lengths:
        results[(name, n_tokens)] = tokens_per_second(fn, n_tokens)
        print(f'{name:30s} {n_tokens:5d} characters: {results[(name, n_tokens)]:7.1f} characters/second', flush=True)

print()
print('| Method | 250 characters | 1,000 characters |')
print('| --- | --- | --- |')
for name, _, _ in rows:
    cells = [f'{results[(name, n)]:.0f}/s' if (name, n) in results else '' for n in (250, 1000)]
    print(f'| {name} | {cells[0]} | {cells[1]} |')
for n in (250, 1000):
    print(f'kv cache against no cache, {n} characters: {results[("kv cache", n)] / results[("no cache (no_grad)", n)]:.1f} times faster')
