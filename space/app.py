# Hugging Face Space: a demo of the Tiny Shakespeare GPT
# the model is small enough to run on the CPU, so nothing here asks for a GPU
try:
    # on a ZeroGPU Space, spaces must be imported before torch
    import spaces
except ImportError:
    spaces = None

import sys
import torch
import gradio as gr

sys.path.insert(0, 'gpt')
# downloads gpt.pt from the GitHub release the first time, then loads it
from checkpoint import load_checkpoint

m, char, encode, decode = load_checkpoint()
vocab = set(char)
n_params = sum(p.numel() for p in m.parameters())

if spaces is not None:
    # ZeroGPU wants at least one GPU function, this one is never called, so no visitor uses GPU quota
    @spaces.GPU(duration=1)
    def _unused():
        pass


def write(prompt: str = 'ROMEO:\n', characters: int = 300, seed: int = 0):
    """Continue a prompt in the style of Shakespeare, one character at a time.

    Args:
        prompt: the start of the text. Only letters, digits 3, space, newline and ! $ & ' , - . : ; ? are known.
        characters: how many new characters to write (50 to 1000).
        seed: the same prompt and seed always give the same text, 0 gives a different text every time.
    """
    unknown = sorted(set(prompt) - vocab)
    if unknown:
        raise gr.Error(f'The model does not know these characters: {" ".join(unknown)}')
    idx = torch.tensor([encode(prompt) or [0]], dtype=torch.long)
    generator = torch.Generator()
    if seed:
        generator.manual_seed(int(seed))
    else:
        generator.seed()
    text = prompt
    # stream: show the text while it is being written
    for i, idx_next in enumerate(m.generate_stream(idx, int(characters), generator=generator)):
        text += decode(idx_next[0].tolist())
        if i % 5 == 4:
            yield text
    yield text


with gr.Blocks(title='Tiny Shakespeare GPT') as demo:
    gr.Markdown(
        f'# Tiny Shakespeare GPT\n'
        f'A {n_params / 1e6:.1f}M parameter character-level transformer, built and trained from scratch. '
        f'It writes one character at a time, with a KV cache, on a CPU. '
        f'Val loss 1.2751, test loss 1.3370 on Shakespeare\'s poems, which it never saw. '
        f'[Code and notes on GitHub](https://github.com/ranmori/Tiny_ai_stack)'
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox('ROMEO:\n', label='Prompt', lines=4)
            characters = gr.Slider(50, 1000, 300, step=10, label='Characters to write')
            seed = gr.Number(0, label='Seed (0 = random)', precision=0)
            button = gr.Button('Write', variant='primary')
        output = gr.Textbox(label='Text', lines=20)
    gr.Examples(
        [['ROMEO:\n'], ['JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n'],
         ['KING RICHARD III:\nNow is the winter of our discontent\n'], ['First Citizen:\n']],
        inputs=[prompt],
    )
    button.click(write, [prompt, characters, seed], output)

demo.launch()
