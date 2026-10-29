# how does the server hold up when several people use it at the same time?
# start the server first (uvicorn server:app --app-dir gpt --port 8000), then from the project root:
#   python gpt/loadtest.py                        against http://127.0.0.1:8000
#   python gpt/loadtest.py http://127.0.0.1:8001  against another address, for example the Docker container
#
# for 1, 2, 4, 8 and 16 users at once, every user streams one request of 200 characters
# measured per request: time to the first character, and the total time
# measured overall: how many characters per second the server makes for everyone together
import asyncio
import statistics
import sys
import time
import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'
N_CHARACTERS = 200


async def one_request(client, i):
    t0 = time.perf_counter()
    first = None
    received = 0
    body = {'prompt': 'ROMEO:\n', 'max_new_tokens': N_CHARACTERS, 'seed': i}
    async with client.stream('POST', URL + '/generate/stream', json=body) as response:
        response.raise_for_status()
        async for chunk in response.aiter_text():
            if chunk and first is None:
                first = time.perf_counter() - t0
            received += len(chunk)
    return first, time.perf_counter() - t0, received


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]


async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        info = (await client.get(URL + '/health')).json()
        print(f'{URL}: {info}')
        await one_request(client, 0) # warm-up
        print()
        print('| Users at once | First character p50 | First character p95 | Total p50 | Total p95 | Characters/second, all users |')
        print('| --- | --- | --- | --- | --- | --- |')
        for users in (1, 2, 4, 8, 16):
            t0 = time.perf_counter()
            results = await asyncio.gather(*[one_request(client, i) for i in range(users)])
            wall = time.perf_counter() - t0
            firsts = [r[0] for r in results]
            totals = [r[1] for r in results]
            chars = sum(r[2] for r in results)
            print(f'| {users} | {statistics.median(firsts):.2f} s | {p95(firsts):.2f} s | '
                  f'{statistics.median(totals):.2f} s | {p95(totals):.2f} s | {chars / wall:.0f} |', flush=True)


asyncio.run(main())
