# make extra training text: Shakespeare's plays from Project Gutenberg, cleaned to look like input.txt
# run from the project root: python gpt/data/prepare.py  (it downloads about 5.6 MB, then writes gpt/data/extra.txt)
#
# the val data in train.py is the last 10% of input.txt: the end of The Taming of the Shrew and The Tempest
# those two plays are left out completely, otherwise the model would see the val text while training
# plays that are already in input.txt are left out too, they would add nothing new
import re
import unicodedata
import urllib.request

URL = 'https://www.gutenberg.org/cache/epub/100/pg100.txt'
SKIP = {'THE TAMING OF THE SHREW', 'THE TEMPEST'}
# not plays: no speakers, so they do not look like input.txt
POEMS = {'THE SONNETS', 'A LOVER’S COMPLAINT', 'THE PASSIONATE PILGRIM', 'THE PHOENIX AND THE TURTLE',
         'THE RAPE OF LUCRECE', 'VENUS AND ADONIS'}

raw = urllib.request.urlopen(URL).read().decode('utf-8')
# only the books themselves, without the Project Gutenberg header and licence
raw = raw[raw.index('*** START'):raw.index('*** END')]
raw = raw[raw.index('\n') + 1:].replace('\r\n', '\n')

with open('gpt/data/input.txt', encoding='utf-8') as f:
    tiny = f.read()
# the 65 characters the model knows, the extra text may only use these
vocab = set(tiny)

# the list of works at the top, every title is indented by 4 spaces
contents = raw[raw.index('Contents'):raw.index('THE SONNETS\n\n')]
titles = [line.strip() for line in contents.split('\n')[1:] if line.strip()]

# cut the book into works: each one starts where its title is a line of its own
starts = sorted((m.start(), t) for t in titles for m in re.finditer('^' + re.escape(t) + '$', raw, re.M))
# the first match of each title is the list of works itself, keep the last one
first = {}
for pos, t in starts:
    first[t] = pos
order = sorted((pos, t) for t, pos in first.items())
works = {t: raw[pos:(order[i + 1][0] if i + 1 < len(order) else len(raw))] for i, (pos, t) in enumerate(order)}

# the same characters, written the way input.txt writes them
REPLACE = {'’': "'", '‘': "'", '“': "'", '”': "'", '—': '--', '–': '-', 'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE'}

def clean_line(line):
    for a, b in REPLACE.items():
        line = line.replace(a, b)
    # stage directions in the middle of a line: [_Embracing_]
    line = re.sub(r'\[.*?\]', '', line)
    # accents: é -> e
    line = unicodedata.normalize('NFKD', line)
    line = ''.join(ch for ch in line if not unicodedata.combining(ch))
    # anything else the model does not know is dropped
    line = ''.join(ch for ch in line if ch in vocab)
    return re.sub(' +', ' ', line).strip()

HEADING = r'^(ACT|SCENE|EPILOGUE|PROLOGUE|THE PROLOGUE|THE EPILOGUE|INDUCTION)\b'
SPEAKER = r"^([A-Z][A-Z ’'\-,&]+)\.$"

def clean_play(text):
    # the play starts at its first speaker, after the list of scenes and the list of characters
    # a speaker is a name in capitals ending in a full stop, with a line of speech straight after it
    # the list of characters looks like speakers too, so jump past it to the first act or scene heading
    if 'Dramatis Person' in text:
        text = text[text.index('Dramatis Person'):]
        text = text[re.search(HEADING, text, re.M).start():]
    lines = text.split('\n')
    start = next(i for i in range(len(lines) - 1)
                 if re.match(SPEAKER, lines[i]) and not re.match(HEADING, lines[i])
                 and lines[i + 1].strip() and not lines[i + 1].startswith(' ')
                 and not re.match(SPEAKER, lines[i + 1]))
    text = '\n'.join(lines[start:])
    # some plays have a different title in the book than in the list of works, so the next play's
    # title and contents can stick to the end of this one, cut them off
    if '\nContents\n' in text:
        text = text[:text.index('\nContents\n')].rstrip()
        text = text[:text.rindex('\n')]
    out = []
    in_direction = False
    for line in text.split('\n'):
        # stage directions that run over several lines: [_Exeunt ... until ..._]
        if in_direction:
            in_direction = ']' not in line
            continue
        if line.lstrip().startswith('[') and ']' not in line:
            in_direction = True
            continue
        # act and scene headings, and the epilogue / prologue headings
        if re.match(HEADING, line):
            continue
        # stage directions on their own line are indented: " Enter Antony."
        if line.startswith(' ') and line.strip():
            continue
        # and some are not indented: "Enter Hamlet."
        if re.match(r'^(Enter|Re-enter|Exit|Exeunt|Flourish|Alarum|Alarums|Sennet|Trumpets|Hautboys)\b', line):
            continue
        # speaker: "CLEOPATRA." -> "CLEOPATRA:"
        m = re.match(SPEAKER, line)
        if m:
            out.append(clean_line(m.group(1)) + ':')
            continue
        out.append(clean_line(line))
    text = '\n'.join(out)
    # at most one empty line in a row, like input.txt
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n\n'

# input.txt comes from a different edition, so the lines are not exactly the same
# (other spelling and punctuation), compare runs of 6 words instead, lower case and letters only
def shingles(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {' '.join(words[i:i + 6]) for i in range(len(words) - 5)}

tiny_shingles = shingles(tiny)

def overlap(text):
    # the share of the play's 6-word runs that also appear in input.txt
    s = shingles(text)
    return len(s & tiny_shingles) / max(1, len(s))

extra = []
for t in titles:
    name = clean_line(t)
    if t in POEMS:
        continue
    if t in SKIP:
        print(f'skip (val data)                  {name}')
        continue
    play = clean_play(works[t])
    o = overlap(play)
    # a play that is not in input.txt still shares a few common 6-word runs by chance, about 1 to 2%
    if o > 0.10:
        print(f'skip (in input.txt, {o:4.0%} same)  {name}')
        continue
    print(f'use  {len(play):8,d} chars, {o:3.0%} same  {name}')
    extra.append(play)

extra = ''.join(extra)
assert set(extra) <= vocab
with open('gpt/data/extra.txt', 'w', encoding='utf-8', newline='\n') as f:
    f.write(extra)
print(f'extra.txt: {len(extra):,} characters, input.txt: {len(tiny):,} characters')

# a test set that nothing was ever tuned on: the poems
# every play is already used for training or val, the poems are the only Shakespeare left that the model never saw
# they are verse without speakers, so the test loss is not comparable with the val loss, it is a check on unseen text
def clean_poem(text):
    out = []
    for line in text.split('\n')[1:]:          # the first line is the title
        if re.fullmatch(r'\s*[0-9IVXLC]+\.?\s*', line):  # sonnet numbers and roman numeral headings
            continue
        out.append(clean_line(line))
    text = '\n'.join(out)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n\n'

test = ''.join(clean_poem(works[t]) for t in titles if t in POEMS)
assert set(test) <= vocab
# the poems must not appear in the training text, check with the same 6-word runs as above
train_shingles = tiny_shingles | shingles(extra)
s = shingles(test)
print(f'test.txt: {len(test):,} characters from {len(POEMS)} poems, '
      f'{len(s & train_shingles) / len(s):.2%} of its 6-word runs also in input.txt or extra.txt')
with open('gpt/data/test.txt', 'w', encoding='utf-8', newline='\n') as f:
    f.write(test)
