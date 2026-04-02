FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System Python 3.10 comes with Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Verify system Python 3.10
RUN python3 --version

# ── Install Miniconda ──
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# ── Create conda environment with Python 3.13.12 ──
COPY docker/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Make conda games env the default shell environment
ENV CONDA_DEFAULT_ENV=games
ENV PATH="${CONDA_DIR}/envs/games/bin:${PATH}"

# ── Install PyTorch (CUDA 12.6) ──
COPY docker/requirements-torch.txt /tmp/requirements-torch.txt
RUN pip install --no-cache-dir -r /tmp/requirements-torch.txt

# torchaudio was built against cu129 in the original env
RUN pip install --no-cache-dir torchaudio==2.9.1+cu129 --extra-index-url https://download.pytorch.org/whl/cu129

# ── Install all other pip dependencies ──
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Copy repo and install local packages ──
COPY . /workspace
WORKDIR /workspace

# tau2-bench
RUN pip install --no-cache-dir -e tau2-bench/

# ToolSandbox (--no-deps: its pyproject.toml pins stale old versions that
# would downgrade numpy, transformers, pydantic, etc.)
RUN pip install --no-cache-dir --no-deps -e evals/benchmarks/ToolSandbox/

# Custom vllm with weighted LoRA support (--no-deps: its pyproject.toml
# requires anthropic>=0.71.0 which conflicts with ToolSandbox's anthropic==0.26.1;
# all actual runtime deps are already installed above)
RUN pip install --no-cache-dir --no-deps -e vllm_weighted_lora/

# ── Shell init: activate conda env on login ──
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate games" >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
