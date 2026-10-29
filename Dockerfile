# the model server in a container, CPU only
#   docker build -t tiny-gpt .
#   docker run -p 8000:8000 tiny-gpt
# the trained model (gpt.pt, about 52 MB) is not in the image: on start it is downloaded from the GitHub release,
# or a local one can be used without downloading:
#   docker run -p 8000:8000 -v "$PWD/gpt/gpt.pt:/app/gpt/gpt.pt" tiny-gpt
FROM python:3.13-slim

WORKDIR /app

# the CPU-only build of PyTorch: about a fifth of the size of the default one, which includes GPU libraries
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# only the code the server needs, not the training data
COPY gpt/model.py gpt/checkpoint.py gpt/batcher.py gpt/server.py gpt/

EXPOSE 8000
CMD ["uvicorn", "server:app", "--app-dir", "gpt", "--host", "0.0.0.0", "--port", "8000"]
