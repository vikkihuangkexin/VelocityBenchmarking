ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

ARG MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
ARG INTERVELO_REPO=https://github.com/sd68515/InterVelo_py311.git
ARG INTERVELO_REF=py311-docker

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libssl-dev \
    libxext6 \
    libxrender1 \
    procps \
    tini \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /opt/InterVelo

RUN wget -qO /tmp/miniforge.sh ${MINIFORGE_URL} && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda config --system --set auto_activate_base false && \
    ${CONDA_DIR}/bin/conda config --system --set channel_priority strict && \
    ${CONDA_DIR}/bin/conda config --system --remove-key channels || true && \
    ${CONDA_DIR}/bin/conda config --system --add channels defaults && \
    ${CONDA_DIR}/bin/conda config --system --add channels conda-forge && \
    ${CONDA_DIR}/bin/conda config --system --set show_channel_urls true && \
    ${CONDA_DIR}/bin/conda install -y -n base mamba

RUN mamba create -y -n intervelo -c conda-forge \
    python=3.11 \
    pip && \
    mamba install -y -n intervelo -c conda-forge \
    numpy=1.26.4 \
    numba=0.59.1 \
    nvidia-ml-py \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    anndata=0.12.10 \
    scanpy=1.11.5 \
    scvelo=0.3.4 && \
    ${CONDA_DIR}/envs/intervelo/bin/python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    ${CONDA_DIR}/envs/intervelo/bin/python -m pip install --no-cache-dir --no-deps torchdiffeq==0.2.2 && \
    ${CONDA_DIR}/envs/intervelo/bin/python -c "import anndata, scanpy, scvelo, torch, torchdiffeq; print('import check ok:', torch.__version__, torch.version.cuda)" && \
    mamba clean -afy

RUN git clone --branch ${INTERVELO_REF} --depth 1 ${INTERVELO_REPO} /opt/InterVelo && \
    ${CONDA_DIR}/envs/intervelo/bin/python -m pip install --no-deps /opt/InterVelo

RUN printf '%s\n' ". ${CONDA_DIR}/etc/profile.d/conda.sh" "conda activate intervelo" >> /etc/bash.bashrc && \
    printf '%s\n' ". ${CONDA_DIR}/etc/profile.d/conda.sh" "conda activate intervelo" >> /root/.bashrc && \
    chmod -R u+rwX ${CONDA_DIR} /opt/InterVelo

ENV PATH=${CONDA_DIR}/envs/intervelo/bin:${PATH}
ENV CONDA_DEFAULT_ENV=intervelo

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
