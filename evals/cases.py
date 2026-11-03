# what the eval suite asks the model to do: the prompts, the sampling settings, and the pass marks
#
# the prompts may only use the 65 characters the model knows, so no curly quotes and no digits but 3.
from graders import is_heading

# ----- free writing: no prompt, just "write a play" -----
# a newline is how train.py and generate.py start the model off: index 0, the start of a fresh line.
FREE_SAMPLES = 8          # how many pieces of text to write
FREE_LENGTH = 600         # characters each, long enough for several speakers to come and go

# ----- the sampling settings to compare -----
# temperature below 1 makes the model play safe, above 1 makes it take risks. top_k cuts the
# unlikeliest characters out altogether. The question each eval answers: what does that cost and buy?
SWEEP = [
    ('greedy (t=0)', dict(temperature=0)),
    ('t=0.5', dict(temperature=0.5)),
    ('t=0.8', dict(temperature=0.8)),
    ('t=1.0 (as released)', dict()),
    ('t=1.0, top_k=10', dict(top_k=10)),
    ('t=1.2', dict(temperature=1.2)),
    ('t=1.5', dict(temperature=1.5)),
]
SWEEP_SAMPLES = 4
SWEEP_LENGTH = 400


# ----- prompt following: one thing per case, checked on the continuation only -----
# each check gets the characters the model added (not the prompt) and says yes or no.
# min_rate is not a target to aim at, it is a line that a working model is well clear of:
# it is there to fail loudly if a future change breaks the behaviour, not to be tuned against.

def speech_first(new):
    # given a speaker heading, the model should say something, not immediately name another speaker
    first = new.split('\n')[0]
    return bool(first.strip()) and not is_heading(first)


def no_newline_first(new):
    # given half a line, the model should finish the line before pressing enter
    return bool(new) and new[0] != '\n'


def heading_first(new):
    # after a blank line, a play starts the next speech with a heading
    return is_heading(new.split('\n')[0])


def keeps_the_verse(new):
    # given a line of verse with no speakers in it, the model should carry on with lines of a
    # similar length rather than dropping back into play format straight away
    lines = [l for l in new.split('\n')[:4] if l]
    return bool(lines) and not any(is_heading(l) for l in lines)


PROMPT_CASES = [
    dict(name='after a heading, speak',
         prompt='ROMEO:\n', length=80, check=speech_first, min_rate=0.9),
    dict(name='finish the line',
         prompt='To be, or not to be, that is the ', length=60, check=no_newline_first, min_rate=0.8),
    dict(name='after a blank line, name a speaker',
         prompt='KING RICHARD III:\nA horse! a horse! my kingdom for a horse!\n\n',
         length=40, check=heading_first, min_rate=0.7),
    # this last one has no pass mark: the model was only ever trained on plays, so there is no
    # reason it should keep writing verse. It is measured because what it does with a style it has
    # not seen is worth knowing, not because a particular number is expected.
    dict(name='carry on with verse (measured, no pass mark)',
         prompt='Shall I compare thee to a summer s day?\nThou art more lovely and more temperate:\n',
         length=120, check=keeps_the_verse, min_rate=None),
]
PROMPT_SAMPLES = 10   # each case is run this many times with different seeds, the result is a rate


# ----- the pass marks for the free writing -----
# the real-text column of the same graders is what these were set against (see evals/notes.md):
# every one is loose enough that the released model is clear of it by a wide margin.
FREE_CHECKS = [
    ('writes speaker headings', lambda s: s['headings'] >= 1, 'at least one heading per sample'),
    ('headings sit after a blank line', lambda s: s['heading_after_blank'] >= 0.9, '>= 90%'),
    ('uses real speaker names', lambda s: s['known_speaker_rate'] >= 0.5, '>= 50% of names are real'),
    ('does not loop', lambda s: s['repeat_rate_8'] <= 0.10, 'repeated 8-runs <= 10%'),
    ('no long copied passage', lambda s: s['longest_repeat'] <= 60, 'longest repeat <= 60 characters'),
    ('writes real words', lambda s: s['real_word_rate'] >= 0.85, '>= 85% of words are in the text'),
]
