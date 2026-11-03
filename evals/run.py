# the eval suite: everything the val loss does not tell us about the model
# run from the project root: python evals/run.py            (about 4 minutes on a laptop CPU)
#                            python evals/run.py --quick    (fewer and shorter samples, about 1 minute)
#                            python evals/run.py --model gpt/gpt_fixed_lr.pt
#
# the loss says how surprised the model is by real Shakespeare. It says nothing about the text the model
# writes itself: whether it keeps the shape of a play, invents its speakers, falls into a loop, or spells
# words that exist. This suite measures those, runs the same graders over real Shakespeare so every number
# has something to be compared with, and writes it all to evals/results.json so two checkpoints can be
# put side by side.
#
# everything here is deterministic: fixed seeds for the sampling, whole texts in fixed windows for the loss.
# Two runs on the same checkpoint give the same numbers.
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, 'gpt')
sys.path.insert(0, 'evals')

import torch

import cases
import graders
from checkpoint import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt/gpt.pt')
parser.add_argument('--out', default='evals/results.json')
parser.add_argument('--seed', type=int, default=1234, help='the sampling seed: another seed shows how much of a difference is noise')
parser.add_argument('--quick', action='store_true', help='fewer, shorter samples: for checking the suite itself')
parser.add_argument('--loss-chars', type=int, default=0, help='only score the first N characters of each text (0 = all)')
args = parser.parse_args()

if args.quick:
    cases.FREE_SAMPLES, cases.FREE_LENGTH = 3, 300
    cases.SWEEP_SAMPLES, cases.SWEEP_LENGTH = 2, 150
    cases.PROMPT_SAMPLES = 4
    args.loss_chars = args.loss_chars or 20_000

started = time.time()
m, char, encode, decode = load_checkpoint(args.model)
block_size = m.block_size
params = sum(p.numel() for p in m.parameters())
print(f'model: {args.model}, {params:,} parameters, block_size {block_size}, '
      f'{torch.get_num_threads()} CPU threads')

# ----- the texts -----
with open('gpt/data/input.txt', encoding='utf-8') as f:
    tiny = f.read()
with open('gpt/data/extra.txt', encoding='utf-8') as f:
    extra = f.read()
with open('gpt/data/test.txt', encoding='utf-8') as f:
    test_text = f.read()
# train.py splits input.txt 90/10 and adds extra.txt to the training side
train_text = tiny[:int(0.9 * len(tiny))] + extra
val_text = tiny[int(0.9 * len(tiny)):]

# what the model could have learnt from: every word and every speaker name in the training text
vocab = graders.vocabulary_of(train_text)
real_speakers = graders.speakers_in(train_text)
print(f'training text: {len(train_text):,} characters, {len(vocab):,} different words, '
      f'{len(real_speakers):,} speaker names')


def table(rows, headers):
    # a markdown table, so the output can go straight into the notes
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: '| ' + ' | '.join(str(c).ljust(w) for c, w in zip(r, widths)) + ' |'
    print(line(headers))
    print('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
    for r in rows:
        print(line(r))
    print()


# ----- 1. loss on held-out text, deterministically -----

@torch.no_grad()
def loss_on(text, batch_size=16):
    # the same measurement as gpt/evaluate.py: windows of block_size characters, one after the other,
    # over the whole text, so there is nothing random in the number
    if args.loss_chars:
        text = text[:args.loss_chars]
    data = torch.tensor(encode(text), dtype=torch.long)
    starts = list(range(0, len(data) - block_size - 1, block_size))
    total = count = 0
    for b in range(0, len(starts), batch_size):
        ix = starts[b:b + batch_size]
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        _, loss = m(x, y)
        total += loss.item() * y.numel()
        count += y.numel()
    return total / count


print('\n## 1. loss on held-out text\n')
losses = {}
rows = []
for name, text, tuned in [('val (end of The Shrew, The Tempest)', val_text, 'yes'),
                          ('test (the poems)', test_text, 'never')]:
    loss = loss_on(text)
    losses[name] = loss
    rows.append([name, f'{len(text):,}', f'{loss:.4f}', f'{graders.bits_per_char(loss):.3f}', tuned])
rows.append(['guessing (1 of 65 characters)', '-', f'{math.log(65):.4f}', f'{math.log2(65):.3f}', '-'])
table(rows, ['Text', 'Characters', 'Loss', 'Bits/char', 'Used for choices?'])


# ----- generating, in batches so the CPU is not left idle -----

def write(prompt, n, length, **sampling):
    # n pieces of text from the same prompt, generated together in one batch (one forward pass per
    # character for all n rows), then handed back as the characters that were added after the prompt.
    # the seed is fixed, so this gives the same text on every run.
    if sampling.get('temperature') == 0:
        n = 1   # greedy takes the most likely character every time: all rows would be the same text
    torch.manual_seed(args.seed)
    start = encode(prompt) or [0]   # an empty prompt starts from a newline, like generate.py
    idx = torch.tensor([start] * n, dtype=torch.long)
    out = m.generate(idx, length, **sampling)
    return [decode(row[len(start):].tolist()) for row in out]


# ----- 2. free writing, against real Shakespeare -----

print(f'\n## 2. free writing: {cases.FREE_SAMPLES} samples of {cases.FREE_LENGTH} characters, '
      f'no prompt, as released (t=1.0)\n')
samples = write('', cases.FREE_SAMPLES, cases.FREE_LENGTH)
scored = [graders.score(s, vocab, real_speakers) for s in samples]
# the same graders over real Shakespeare, in pieces of the same size, from the held-out val text:
# this is the column that says what the numbers mean
real_pieces = [val_text[i:i + cases.FREE_LENGTH]
               for i in range(0, cases.FREE_SAMPLES * cases.FREE_LENGTH, cases.FREE_LENGTH)]
real_scored = [graders.score(s, vocab, real_speakers) for s in real_pieces]

mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
METRICS = [
    ('headings per sample', 'headings', '{:.2f}'),
    ('headings after a blank line', 'heading_after_blank', '{:.1%}'),
    ('speech lines per heading', 'speech_lines_per_heading', '{:.2f}'),
    ('mean speech line length', 'mean_speech_line_length', '{:.1f}'),
    ('lines ending in a colon that are not headings', 'broken_colon_lines', '{:.2f}'),
    ('speaker names that are real', 'known_speaker_rate', '{:.1%}'),
    ('repeated 8-character runs', 'repeat_rate_8', '{:.2%}'),
    ('longest repeated piece (characters)', 'longest_repeat', '{:.1f}'),
    ('words that are in the training text', 'real_word_rate', '{:.2%}'),
]
table([[label, fmt.format(mean(scored, key)), fmt.format(mean(real_scored, key))]
       for label, key, fmt in METRICS],
      ['Measured over the whole sample set', 'The model', 'Real Shakespeare'])

invented = [w for s in scored for w in s['invented_words']]
print(f'words it made up (a character-level model spells one character at a time): '
      f'{", ".join(invented[:14]) if invented else "none"}\n')
print('the first sample, cut to 300 characters:\n')
print('    ' + samples[0][:300].replace('\n', '\n    '))


# ----- 3. what the sampling settings cost and buy -----

print(f'\n\n## 3. sampling settings: {cases.SWEEP_SAMPLES} samples of {cases.SWEEP_LENGTH} characters each\n')
sweep = {}
rows = []
for label, sampling in cases.SWEEP:
    texts = write('', cases.SWEEP_SAMPLES, cases.SWEEP_LENGTH, **sampling)
    s = [graders.score(t, vocab, real_speakers) for t in texts]
    sweep[label] = {k: mean(s, k) for _, k, _ in METRICS}
    rows.append([label,
                 f'{mean(s, "real_word_rate"):.1%}',
                 f'{mean(s, "repeat_rate_8"):.1%}',
                 f'{mean(s, "longest_repeat"):.0f}',
                 f'{mean(s, "known_speaker_rate"):.0%}',
                 f'{mean(s, "headings"):.1f}'])
real = {k: mean(real_scored, k) for _, k, _ in METRICS}
rows.append(['real Shakespeare',
             f'{real["real_word_rate"]:.1%}', f'{real["repeat_rate_8"]:.1%}',
             f'{real["longest_repeat"]:.0f}', f'{real["known_speaker_rate"]:.0%}', '-'])
table(rows, ['Setting', 'Real words', 'Repeated 8-runs', 'Longest repeat', 'Real names', 'Headings'])


# ----- 4. does it do what the prompt asks? -----

print(f'\n## 4. following the prompt: every case run {cases.PROMPT_SAMPLES} times\n')
prompts = {}
rows = []
for case in cases.PROMPT_CASES:
    texts = write(case['prompt'], cases.PROMPT_SAMPLES, case['length'])
    rate = sum(1 for t in texts if case['check'](t)) / len(texts)
    prompts[case['name']] = rate
    if case['min_rate'] is None:
        rows.append([case['name'], repr(case['prompt'][-28:]), f'{rate:.0%}', '-', 'measured'])
    else:
        rows.append([case['name'], repr(case['prompt'][-28:]), f'{rate:.0%}',
                     f'>= {case["min_rate"]:.0%}', 'pass' if rate >= case['min_rate'] else 'FAIL'])
table(rows, ['Case', 'Prompt ends with', 'Rate', 'Pass mark', ''])


# ----- 5. the pass marks -----
# these are regression guards: lines a working model is well clear of, so a change that breaks
# the model's writing fails here instead of looking fine because the loss barely moved.

print('\n## 5. pass marks\n')
checks = []
for label, check, wanted in cases.FREE_CHECKS:
    rates = [check(s) for s in scored]
    rate = sum(rates) / len(rates)
    # the free-writing checks must hold for every sample
    checks.append((label, wanted, rate == 1, f'{rate:.0%} of samples'))
for case in cases.PROMPT_CASES:
    if case['min_rate'] is None:
        continue   # measured, not a pass mark
    rate = prompts[case['name']]
    checks.append((f'prompt: {case["name"]}', f'>= {case["min_rate"]:.0%}',
                   rate >= case['min_rate'], f'{rate:.0%}'))
table([[label, wanted, got, 'pass' if ok else 'FAIL'] for label, wanted, ok, got in checks],
      ['Check', 'Pass mark', 'Measured', ''])

failed = [label for label, _, ok, _ in checks if not ok]
minutes = (time.time() - started) / 60


# ----- 6. write it all down -----

commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                        capture_output=True, text=True).stdout.strip() or 'unknown'
results = {
    'model': args.model,
    'model_bytes': os.path.getsize(args.model),
    'parameters': params,
    'commit': commit,
    'seed': args.seed,
    'quick': args.quick,
    'minutes': round(minutes, 2),
    'loss': {name: round(loss, 4) for name, loss in losses.items()},
    'bits_per_char': {name: round(graders.bits_per_char(loss), 4) for name, loss in losses.items()},
    'free_writing': {
        'samples': cases.FREE_SAMPLES, 'length': cases.FREE_LENGTH,
        'model': {k: mean(scored, k) for _, k, _ in METRICS},
        'real_shakespeare': real,
    },
    'sampling_sweep': sweep,
    'prompt_cases': prompts,
    'checks': {label: ok for label, _, ok, _ in checks},
    'failed': failed,
}
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f'{len(checks) - len(failed)} of {len(checks)} checks passed in {minutes:.1f} minutes, '
      f'written to {args.out}')
if failed:
    print('failed: ' + ', '.join(failed))
if args.quick:
    # --quick is for checking that the suite itself still runs. Three short samples are too few for
    # the pass marks to mean anything, so a failure there is not reported as one.
    print('--quick: samples are too short and too few to judge the model by, numbers are indicative only')
elif failed:
    sys.exit(1)
