# a small web server around the trained model
# run from the project root: uvicorn server:app --app-dir gpt --port 8000
# BATCHING=0 switches request batching off (see batcher.py), it is on by default
#
#   GET  /health           is the model loaded, and what is it
#   POST /generate         {"prompt": "ROMEO:\n", "max_new_tokens": 200, "seed": 1} -> the whole text at once
#   POST /generate/stream  the same request, but the characters arrive one by one while they are being made
#
# the endpoints are plain def, not async def: generating keeps the CPU busy, and FastAPI runs plain def
# endpoints in a pool of threads, so one long request does not block the server from answering others
import os
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from batcher import Batcher
from checkpoint import load_checkpoint

# load the model once when the server starts, not on every request
m, char, encode, decode = load_checkpoint()
vocab = set(char)

# with batching one worker thread runs all requests together, without it every request runs on its own thread
batching = os.environ.get('BATCHING', '1') != '0'
batcher = Batcher(m) if batching else None

app = FastAPI(title='Tiny AI Stack GPT')


class GenerateRequest(BaseModel):
    # empty prompt: start from a newline, like generate.py
    prompt: str = ''
    max_new_tokens: int = Field(200, ge=1, le=1000)
    # the same seed and prompt give the same text, no seed gives a different text every time
    seed: int | None = None


def prepare(req):
    # the model only knows 65 characters, anything else can not be encoded
    unknown = sorted(set(req.prompt) - vocab)
    if unknown:
        raise HTTPException(400, f'the prompt has characters the model does not know: {unknown}')
    idx = torch.tensor([encode(req.prompt) or [0]], dtype=torch.long)
    # every request gets its own random generator, so requests running at the same time do not share one
    generator = torch.Generator()
    if req.seed is not None:
        generator.manual_seed(req.seed)
    else:
        generator.seed()
    return idx, generator


@app.get('/health')
def health():
    info = {'status': 'ok', 'parameters': sum(p.numel() for p in m.parameters()),
            'block_size': m.block_size, 'characters': len(char), 'batching': batching}
    if batching and batcher.batch_sizes:
        info['average_batch_size'] = round(sum(batcher.batch_sizes) / len(batcher.batch_sizes), 2)
    return info


def new_characters(idx, max_new_tokens, generator):
    # the new characters one by one, through the batcher or straight from the model
    if batching:
        request = batcher.submit(idx[0].tolist(), max_new_tokens, generator)
        while (token := request.out.get()) is not None:
            if isinstance(token, Exception):
                raise token
            yield decode([token])
    else:
        for idx_next in m.generate_stream(idx, max_new_tokens, generator=generator):
            yield decode(idx_next[0].tolist())


@app.post('/generate')
def generate(req: GenerateRequest):
    idx, generator = prepare(req)
    t0 = time.perf_counter()
    text = ''.join(new_characters(idx, req.max_new_tokens, generator))
    seconds = time.perf_counter() - t0
    return {'text': text, 'characters': req.max_new_tokens,
            'seconds': round(seconds, 3), 'characters_per_second': round(req.max_new_tokens / seconds, 1)}


@app.post('/generate/stream')
def generate_stream(req: GenerateRequest):
    # check the prompt before the response starts, so a bad prompt still gets a proper 400
    idx, generator = prepare(req)

    return StreamingResponse(new_characters(idx, req.max_new_tokens, generator), media_type='text/plain')
