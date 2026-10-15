

from model import GPTLanguageModel
from plots import plot_attention_training

with open("gpt/data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(text[:1000])
print("Text length:", len(text))

# encode and decode 
# all the unique characters are
# remove the duplicates  and then sort into a list
char = sorted(list(set(text)))
vocab_size= len(char)
print('' .join(char))
print(vocab_size)

# mapping all the chars into integers 
# first vocab mapping 
# each alphabet gets a single corresponding integer
# LLMs don't see text — they see numbers.

# stoi= {} # an emtpy dic 
# for i,ch in enumerate(char): #the loop 
#     stoi[ch]= i #add one key value pair each time 

stoi={ch:i for i, ch in enumerate(char)}
# stoi['h'] might be 46.
# the reverse dictionary 
itos={i:ch for i,ch in enumerate(char)}
# itos[46] gives you back 'h'

# then encoding
# a lambda function that the takes s and returns whatever expression in there basically a nameless function
# # lambda  s: <expression>
# def encode(s):
#     return [stoi[c] for c in s]
#  for c in s — loop over the string s, one character at a time. First c = 'h', then c = 'i', etc.
# stoi[c] — for each character, look it up in the stoi dict and grab its number
# [ ... ] — collect every result into a list
encode = lambda s: [stoi[c] for c in s]
decode= lambda l: ''.join([itos[i] for i in l ])

print(encode('hi there '))
print(decode(encode("hii there")))
                    
# encode the whole dataset 

import torch 
data= torch.tensor(encode(text), dtype= torch.long)
print(data.shape, data.dtype)
print(data[:1000])


n= int(0.9*len(data))
train_data=data[:n]
val_data= data[n:]
    
# lets check the first 9 char
block_size = 8
train_data[:block_size+1]



x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


# i need the same manual seed every time basially the same random ones
torch.manual_seed(1337)
# let my batch size be and the block be 8

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # if the string flag is train then train else val
    data = train_data if split== 'train' else val_data
    # basically randint (random integers )takes a high number and also the size 
    # the reason there is a block size is because the the upper bound keeps every window fully inside the data.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # then pile the data into a 2d tensor stack 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y 


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")


print(xb) # our input to the transformer





# the model sizes, picked by measuring one change at a time (see "Scaling up" in notes.md)
# the examples above used a context of 8 to keep the printing short, the model sees 64 characters at once
block_size = 64
# how many numbers describe each character inside the model
n_embd = 64
# how many attention heads work side by side, n_embd has to divide evenly by it
n_head = 4
# how many blocks (attention then feed forward) are stacked on top of each other, a 4th one barely helped
n_layer = 3
# the share of numbers that dropout sets to zero while training, 0.2 is 20%
# switched off: at this size and number of steps it made the loss worse without closing the train/val gap
dropout = 0.0

m = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
# numel is the number of elements in a tensor, summed over every parameter it is the size of the model
print('number of parameters:', sum(p.numel() for p in m.parameters()))
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# for the plots: one fixed chunk of text, to watch who listens to whom before and after training
# only the first 8 characters, a 64 by 64 grid with the numbers written in would be unreadable
example = train_data[:8]
print('example chunk:', repr(decode(example.tolist())))

# run the example through the model and return the attention weights every head used
# a list with one entry per block, each entry a list with one (T, T) grid per head
@torch.no_grad()
def attention_weights():
    # eval mode so dropout is switched off while looking
    m.eval()
    # unsqueeze(0) adds a batch dimension in front, (T) -> (1, T), because the model expects a batch
    m(example.unsqueeze(0))
    m.train()
    return [[h.wei[0].clone() for h in block.sa.heads] for block in m.blocks]

wei_before = attention_weights()

# what the model writes before any training: random characters
# the starting context is one newline character (index 0)
context = torch.zeros((1, 1), dtype=torch.long)
print('----- text before training -----')
# eval mode so dropout is switched off while writing, then back to train mode
m.eval()
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
m.train()

# optimizer
# create a PyTorch optimizer
# bsically optimization of the model 
# m.paramaters are the weight and biases together and then there is the learning rate which for a small model 
# adamW is doing regulariztion
# AdamW is Adam with decoupled weight decay. Adam keeps a running average of each parameter's gradient and squared gradient, 
# so every weight gets its own adaptive step size.
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# for the plots: the loss at every step
losses = []


# each loop is gonna train on 32 batch
batch_size = 32

# how many batches to average over when estimating the loss
eval_iters = 200

# the loss of one batch is jumpy because it depends on which 32 chunks were picked
# so average the loss over many batches, for both the train data and the val data
# no_grad tells pytorch not to remember anything for backward, we are only measuring here so it is faster
@torch.no_grad()
def estimate_loss():
    out = {}
    # eval mode: some layers behave differently when measuring than when training
    # dropout is switched off here, so the loss is measured with the whole model
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        # the average of the 200 losses for this split
        out[split] = losses.mean().item()
    # back to train mode
    m.train()
    return out

# for the plots: the step, train loss and val loss of every estimate
eval_steps, eval_train, eval_val = [], [], []

# gradient descent
#
for steps in range(10000): # increase number of steps for good results...

    # every 1000 steps measure the loss on train and val
    # if train keeps going down but val goes up the model is memorizing (overfitting)
    if steps % 1000 == 0:
        est = estimate_loss()
        print(f"step {steps}: train loss {est['train']:.4f}, val loss {est['val']:.4f}")
        eval_steps.append(steps)
        eval_train.append(est['train'])
        eval_val.append(est['val'])

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    # raw, unnormalized scores over the vocabulary
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# print the loss .item is making it into a into a plain Python float
print(loss.item())

# the final loss on train and val after all the steps
est = estimate_loss()
print(f"final: train loss {est['train']:.4f}, val loss {est['val']:.4f}")
eval_steps.append(len(losses))
eval_train.append(est['train'])
eval_val.append(est['val'])

# what the model writes after training, from the same starting context
print('----- text after training -----')
m.eval()
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
m.train()

# save the trained model so generate.py can use it without training again
# state_dict is every trained number in the model, char is needed to rebuild the same decode
# this goes before the plots because the plot window blocks the script until it is closed
# settings are saved too, generate.py has to rebuild a model of exactly the same shape before it can load the numbers
settings = {'n_embd': n_embd, 'block_size': block_size, 'n_head': n_head, 'n_layer': n_layer, 'dropout': dropout}
torch.save({'model_state_dict': m.state_dict(), 'char': char, 'settings': settings}, 'gpt/gpt.pt')


# plot the results, the drawing itself lives in plots.py
# two windows: the attention weights of every block and head before training,
# and the loss (with the bigram's val loss as the line to beat) next to the attention weights after training
wei_after = attention_weights()
plot_attention_training(losses, vocab_size, eval_steps, eval_train, eval_val,
                        wei_before, wei_after, decode(example.tolist()), baseline=2.50)
