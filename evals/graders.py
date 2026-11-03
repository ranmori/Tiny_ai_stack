# the measurements the eval suite makes, one function per thing being measured
#
# every grader here takes a piece of text and gives back plain numbers. None of them look at the model,
# so the same grader can score the model's text and real Shakespeare, and the real number says what
# "good" looks like. A grader on its own means very little: 3% repeated runs of 8 characters sounds bad
# until the real plays turn out to be at 2.6%.
import math
import re

# a speaker heading in this dataset is its own line, ending in a colon: "ROMEO:", "First Citizen:", "QUEEN MARGARET:"
NAME_LINE = re.compile(r"^[A-Z][A-Za-z' ]*:$")


def is_heading(line):
    # the colon alone is not enough: plenty of speech lines end in one ("Whereof I know she is not
    # ignorant:"), and they would be counted as speaker names. Real names are short, so a line of more
    # than three words is not a heading. Measured on the training text, this rule finds 7,250 headings
    # in input.txt and 24,146 in extra.txt, and 99.5% / 99.9% of them sit right after a blank line,
    # which is the shape a heading is supposed to have. Without the word count it was 91%, because of
    # all the speech lines ending in a colon.
    return bool(NAME_LINE.match(line)) and len(line.split()) <= 3
WORD = re.compile(r"[A-Za-z']+")


def lines_of(text):
    return text.split('\n')


# ----- 1. loss and bits per character -----

def bits_per_char(loss):
    # cross entropy is in nats (natural log). Divided by ln 2 it is bits: how many yes/no questions
    # it would take on average to pin down the next character. 65 characters guessed blind is log2(65) = 6.02 bits.
    return loss / math.log(2)


# ----- 2. does the text keep the shape of a play? -----

def format_stats(text):
    # input.txt has one shape all through: a speaker heading on its own line, the speech under it,
    # then one blank line before the next heading. These numbers say how often that shape holds.
    lines = lines_of(text)
    headings = [i for i, line in enumerate(lines) if is_heading(line)]
    # a heading should have a blank line (or the start of the text) in front of it
    well_placed = sum(1 for i in headings if i == 0 or lines[i - 1] == '')
    speech = [line for i, line in enumerate(lines) if line and i not in set(headings)]
    # lines that end in a colon but are not a clean heading: "ROMEO and JULIET:", "my lord:" and such
    colon_lines = [line for line in lines if line.endswith(':') and not is_heading(line)]
    return {
        'headings': len(headings),
        'heading_after_blank': len(headings) and well_placed / len(headings),
        'speech_lines_per_heading': len(headings) and len(speech) / len(headings),
        'mean_speech_line_length': len(speech) and sum(len(l) for l in speech) / len(speech),
        'broken_colon_lines': len(colon_lines),
        'heading_names': [lines[i][:-1] for i in headings],
    }


def known_speaker_rate(heading_names, real_speakers):
    # the model invents characters. That is not wrong in itself, but a model that never gets a real
    # speaker name out is copying the shape of a heading without having learnt who is in the plays.
    if not heading_names:
        return 0.0
    return sum(1 for name in heading_names if name in real_speakers) / len(heading_names)


def speakers_in(text):
    # every name used as a heading anywhere in a text, to check generated names against
    return {line[:-1] for line in lines_of(text) if is_heading(line)}


# ----- 3. is it repeating itself? -----

def repeat_rate(text, n=8):
    # the share of n-character runs that have been seen before in the same text.
    # a model falling into a loop ("and the and the and the") pushes this up fast.
    if len(text) <= n:
        return 0.0
    runs = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1 - len(set(runs)) / len(runs)


def longest_repeat(text):
    # the length of the longest piece of text that appears at least twice.
    # binary search on the length: if some run of length L appears twice, so does one of length L - 1,
    # so the answer can be found in log2(len) passes instead of trying every length.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        seen = set()
        found = False
        for i in range(len(text) - mid + 1):
            run = text[i:i + mid]
            if run in seen:
                found = True
                break
            seen.add(run)
        if found:
            low = mid
        else:
            high = mid - 1
    return low


# ----- 4. are the words real words? -----

def word_stats(text, vocab):
    # a character-level model has never been told where a word ends: it spells one character at a time.
    # this counts how many of the words it produced are words that exist in the training text.
    words = [w.lower() for w in WORD.findall(text)]
    if not words:
        return {'words': 0, 'real_word_rate': 0.0, 'invented': []}
    invented = [w for w in words if w not in vocab]
    return {
        'words': len(words),
        'real_word_rate': 1 - len(invented) / len(words),
        'invented': invented[:12],
    }


def vocabulary_of(*texts):
    # every word in the training text, lower case, for word_stats to check against
    vocab = set()
    for text in texts:
        vocab.update(w.lower() for w in WORD.findall(text))
    return vocab


# ----- 5. everything at once -----

def score(text, vocab, real_speakers):
    # all the graders on one piece of text, as one flat dict, so the model's text and real
    # Shakespeare go through exactly the same code
    stats = format_stats(text)
    words = word_stats(text, vocab)
    return {
        'characters': len(text),
        'headings': stats['headings'],
        'heading_after_blank': stats['heading_after_blank'],
        'speech_lines_per_heading': stats['speech_lines_per_heading'],
        'mean_speech_line_length': stats['mean_speech_line_length'],
        'broken_colon_lines': stats['broken_colon_lines'],
        'known_speaker_rate': known_speaker_rate(stats['heading_names'], real_speakers),
        'repeat_rate_8': repeat_rate(text, 8),
        'longest_repeat': longest_repeat(text),
        'words': words['words'],
        'real_word_rate': words['real_word_rate'],
        'invented_words': words['invented'],
    }
