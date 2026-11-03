# Eval notes

Notes on the `evals` folder: what it measures, why the loss was not enough on its own, and what the
numbers came out as.

| File | What it is |
| --- | --- |
| `graders.py` | The measurements. Each one takes a piece of text and gives back plain numbers, so the same code scores the model's writing and real Shakespeare |
| `cases.py` | The prompts, the sampling settings to compare, and the pass marks |
| `run.py` | Runs all of it, prints the tables below and writes `results.json` |
| `results.json` | The last run's numbers, for comparing two checkpoints |

```bash
python evals/run.py                              # about 2.5 minutes on a laptop CPU
python evals/run.py --quick                      # about 40 seconds, for checking the suite itself
python evals/run.py --model gpt/gpt_fixed_lr.pt  # score a different checkpoint
python evals/run.py --seed 7                     # another sampling seed, to see what is noise
```

## Where the project had got to

Everything so far was measured with one number, cross entropy on held-out text:

| Model | Parameters | Val | Test (poems) |
| --- | --- | --- | --- |
| Guessing 1 of 65 characters | 0 | 4.17 | |
| Best possible bigram | 4,225 | 2.46 | |
| Full transformer, 3 blocks | 42,369 | 2.06 | |
| Scaled up on a laptop CPU | 161,985 | 1.66 | |
| Tutorial size on a GPU, 10,000 steps, fixed 3e-4 | 10.8 M | 1.2958 | 1.3566 |
| **The released model**: 6e-4 start, cosine decay | 10.8 M | **1.2751** | **1.3370** |

That is 1.84 bits per character against 6.02 for guessing. Then the model was made usable: a KV cache
(3.8 to 5.5 times faster generation), a FastAPI server with streaming, and request batching (4.5 times
the throughput at 16 users).

## Why the loss was not enough

Cross entropy asks one question: *how surprised is the model by text somebody else wrote?* It never
looks at the text the model writes itself. Two things follow from that.

**It cannot see a failure that only shows up when generating.** The clearest example is in the
sampling table below: taking the most likely character every time (temperature 0) gives a model with
exactly the same loss text that repeats itself almost beyond use — 83% of its 8-character runs have
been seen before in the same sample, with one 308-character passage said twice. The loss is blind to it
because greedy decoding is a choice made after training.

**A number with nothing beside it does not mean anything.** "1.7% of its 8-character runs are
repeats" sounds like a fault until the real plays turn out to be at 3.0%. So every grader is also run
over real held-out Shakespeare, cut into pieces of the same size, and both columns are always printed
together. The real column is the only reason the model's column can be read at all.

## What is measured

All of it is deterministic: fixed seeds for the sampling, whole texts in fixed windows for the loss.
Two runs on the same checkpoint give the same numbers.

- **Loss and bits per character** on val and on the poems test set, exactly as `gpt/evaluate.py` does it.
- **Free writing**: 8 samples of 600 characters with no prompt, as the released model is served
  (temperature 1.0), each scored on the shape of a play (speaker headings, where they sit, speech lines
  per heading, line length), on whether the speaker names are real, on repetition, and on whether the
  words exist in the training text.
- **Sampling settings**: the same graders at temperature 0, 0.5, 0.8, 1.0, 1.2, 1.5 and at top_k 10,
  to see what each setting costs and buys. This needed `temperature` and `top_k` arguments in
  `model.generate` and `generate_stream`; the defaults (1.0, no top_k) sample from the model's own
  odds, exactly as before.
- **Following the prompt**: four cases, each run 10 times, each checking one thing about the
  continuation only.
- **Pass marks**: lines a working model is well clear of, so that a change which breaks the writing
  fails loudly instead of looking fine because the loss barely moved.

### Defining a speaker heading took two tries

The first rule was "a line that starts with a capital and ends with a colon". On real Shakespeare it
said only 91% of headings sit after a blank line, which is wrong — the dataset is strict about that.
The rule was counting speech lines as headings: *"Whereof I know she is not ignorant:"* is a line
that starts with a capital and ends with a colon. Real names are short, so lines of more than three
words are now excluded. That finds 7,250 headings in `input.txt` and 24,146 in `extra.txt`, and 99.5%
and 99.9% of them sit after a blank line.

**A grader needs checking against known-good text before it is used to judge anything.** The model
was never the problem in that measurement.

## Results: the released model (`gpt.pt`, val 1.2751)

8 samples of 600 characters, temperature 1.0, seed 1234. The right-hand column is real held-out
Shakespeare through the same graders.

| | The model | Real Shakespeare |
| --- | --- | --- |
| Headings per sample | 5.75 | 3.75 |
| Headings after a blank line | 99.0% | 100.0% |
| Speech lines per heading | 3.22 | 5.43 |
| Mean speech line length | 39.7 | 37.7 |
| Lines ending in a colon that are not headings | 0.12 | 1.25 |
| Speaker names that are real | 95.1% | 100.0% |
| Repeated 8-character runs | 1.69% | 2.99% |
| Longest repeated piece | 11.1 characters | 14.1 |
| Words that are in the training text | 97.74% | 98.99% |

- **The format is learnt.** 99% of headings sit after a blank line, and 95% of the names it uses are
  real characters from the plays. A model that had only learnt the *look* of a heading would invent
  most of its names.
- **It spells.** 97.7% of its words exist in the training text, from a model that has never been told
  where a word ends — it writes one character at a time. Real Shakespeare scores 99.0% against the
  same word list (the last 1% is the words the val text uses and the training text does not).
  What is left is close misses: `enswer'd`, `teached`, `marria`, `bonfir`.
- **It does not loop.** Fewer repeated runs than the real plays, in fact, and no copied passage
  longer than 11 characters, so it is writing rather than reciting.
- **It changes speaker too often.** 3.2 speech lines per heading against 5.4 in the real text: the
  speeches are short, so a page of it reads busier than a real scene. This is the clearest gap the
  suite found, and no loss number would have shown it.

## Results: what the sampling setting does

4 samples of 400 characters each.

| Setting | Real words | Repeated 8-runs | Longest repeat | Real names |
| --- | --- | --- | --- | --- |
| Greedy (temperature 0) | 100.0% | **82.7%** | **308** | 100% |
| 0.5 | 99.4% | 11.5% | 18 | 75% |
| 0.8 | 98.7% | 2.9% | 12 | 94% |
| **1.0 (as released)** | 96.5% | 1.1% | 10 | 100% |
| 1.0, top_k 10 | 97.6% | 1.6% | 10 | 62% |
| 1.2 | 92.5% | 0.0% | 7 | 69% |
| 1.5 | 87.3% | 0.4% | 8 | 62% |
| Real Shakespeare | 99.0% | 3.0% | 14 | 100% |

- **Greedy decoding is unusable**, and the loss cannot tell: 83% repeated runs and the same
  308-character passage twice in 400 characters. Always taking the most likely character walks the
  model into a loop it has no way out of.
- **Temperature trades spelling against repetition**, and it trades it steeply. Every step up buys
  fewer repeats and costs real words: 99.4% at 0.5, 96.5% at 1.0, 87.3% at 1.5.
- **0.8 is the closest to the real text** on both at once (98.7% real words, 2.9% repeats against the
  real 99.0% and 3.0%), which is a reason to try it as the server's default. That is a choice made on
  these numbers, so it would need saying out loud if it is made.

## Results: following the prompt

Each case 10 times, temperature 1.0.

| Case | Rate | Pass mark |
| --- | --- | --- |
| After `ROMEO:`, say something instead of naming another speaker | 100% | >= 90% |
| Given half a line, finish the line before pressing enter | 100% | >= 80% |
| After a blank line, name a speaker | 100% | >= 70% |
| Given two lines of a sonnet, carry on in verse | 70% | measured, no pass mark |

The last one has no pass mark on purpose. The model was only ever trained on plays, so there is no
reason it should keep writing verse; 3 times in 10 it drops straight back into play format with a
speaker heading. Worth knowing, not a fault.

## Results: the suite against two checkpoints

The same suite on `gpt_fixed_lr.pt` (fixed 3e-4, val 1.2958) and on the released model, and the
released model again with a different sampling seed, to see which differences are real:

| | Fixed 3e-4 | Released | Released, seed 7 |
| --- | --- | --- | --- |
| Val loss | 1.2958 | 1.2751 | 1.2751 |
| Test loss (poems) | 1.3566 | 1.3370 | 1.3370 |
| Words in the training text | 96.11% | 97.74% | 97.59% |
| Speaker names that are real | 91.5% | 95.1% | 98.6% |
| Repeated 8-character runs | 1.43% | 1.69% | 2.36% |
| Speech lines per heading | 6.44 | 3.22 | 6.42 |

- **Only one behavioural difference is bigger than the noise.** Changing the sampling seed moves
  word-realness by 0.15 points, so the 1.5-point gap in real words between the two checkpoints is
  real: the model with the better loss also spells better.
- **Everything else is within the noise, and some of it swamps the difference.** Speech lines per
  heading came out 3.22 and 6.42 on the *same* checkpoint with two seeds, so the 3.22-against-6.44
  "finding" above says nothing about the checkpoints. 8 samples of 600 characters is about 4,800
  characters, far too few for the heading counts to settle.
- **So the honest reading:** the better learning rate improved the loss (val and test by the same
  0.02, already recorded) and spelling. It was not shown to improve anything else.

## Results: do the pass marks catch a worse model?

A pass mark nothing has ever failed is not a check. The small laptop-CPU model (`gpt_cpu.pt`, 161,985
parameters, val 1.6594, test 1.6975 — 2.39 bits per character against 1.84) went through the same
suite:

| | Small CPU model | Released model | Real Shakespeare |
| --- | --- | --- | --- |
| Val loss | 1.6594 | 1.2751 | |
| Words in the training text | **82.01%** (fails, mark is 85%) | 97.74% | 98.99% |
| Speaker names that are real | 76.2% | 95.1% | 100.0% |
| Headings after a blank line | 100.0% | 99.0% | 100.0% |
| Speech lines per heading | 5.59 | 3.22 | 5.43 |
| Repeated 8-character runs | 0.27% | 1.69% | 2.99% |
| All four prompt cases | same as the released model | | |

- **One mark out of nine caught it**, and it is the one that matches what the text looks like when
  read: the small model's words are mostly not words (`merchance`, `theyir`, `pimbling`,
  `dissignations`). 82% against 98% is the difference between text that reads as English and text that
  does not.
- **The other marks are too loose to tell the two models apart**, and the prompt cases do not separate
  them at all. Both get the shape of a play right; a 0.4-loss gap is spent almost entirely on
  spelling, not on structure.
- **The structural metrics are worse than useless here**: the small model is *closer* to the real text
  on speech lines per heading (5.59 against 5.43) than the released model's seed-1234 run. Combined
  with the seed noise above, the fair conclusion is that 8 samples cannot measure structure at all.

## Mistakes and things that caught me out

- **The heading rule matched speech lines** (above). Found only because real Shakespeare was scored
  with the same grader and came out at 91% on something that should be 100%.
- **Pass marks on three short samples.** `--quick` cuts to 3 samples of 300 characters, which was
  not long enough for a heading to appear in every sample, so four checks "failed" on a model that is
  fine. `--quick` now says it is for checking the suite, not the model, and does not fail the run.
- **A pass mark the model was never meant to meet.** The verse case started with a 50% mark, failed at
  25%, and the mark was the thing that was wrong: nothing in the training text asks for verse without
  speakers. It is now measured and not marked.
- **Sampling 8 samples one after the other wasted the CPU.** All 8 start from the same single
  character, so they are generated as one batch of 8: one forward pass per character for all of them.
  The whole suite takes 2.5 minutes instead of about 8.

## Still to fix

- **Eight of the nine pass marks have never failed**, even on a model with 0.4 more loss. Only
  word-realness discriminates. The rest are regression guards against something breaking badly
  (a loop, headings in the wrong place), which is worth having, but they are not a quality measure.
- **8 samples is too few** for the structural metrics, which the seed comparison and the small model
  both show. They need something like 50 samples, which is 25 minutes on the laptop CPU, or a GPU.
- **No held-out prompt set.** The four prompt cases were written by hand, and the second one is the
  most famous line in English, which the training text contains. A prompt taken from the poems (never
  trained on) would be a fairer test of following a prompt.

## Next steps

1. Try temperature 0.8 as the server's default, and report both settings' numbers rather than
   quietly changing it.
2. More samples for the structural metrics, so "it changes speaker too often" can be confirmed or
   dropped. On today's numbers it is unconfirmed.
3. A grader that measures something the structural ones miss and the loss does not already cover.
   Word-realness is currently doing all the work; grammar inside a line, or whether a speech answers
   the one before it, is the sort of thing a bigger model would be expected to win on.
