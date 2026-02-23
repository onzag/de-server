# Use Python 3.10.12 slim image as the base
FROM nvidia/cuda:12.9.1-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.9/compat/

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.15.0" --extra-index-url https://download.pytorch.org/whl/cu129

# Copy the requirements file
COPY requirements.txt /requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy your handler code
COPY handler.py /handler.py
COPY base.py /base.py

# Command to run when the container starts
CMD [ "python", "-u", "/handler.py" ]