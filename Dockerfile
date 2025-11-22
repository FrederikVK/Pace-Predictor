FROM mcr.microsoft.com/devcontainers/python:3.11
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true
CMD ["bash"]