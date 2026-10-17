FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CONDA_DEFAULT_ENV=topovelo \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000
ARG TOPOVELO_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl git bzip2 sudo openssh-server vim \
    build-essential libgomp1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /run/sshd /data/simdata /home/${USERNAME}/.ssh && \
    chown -R ${USERNAME}:${USERNAME} /data/simdata /home/${USERNAME}/.ssh && \
    chmod 700 /home/${USERNAME}/.ssh

# Log in as the benchmark user with the standard password scheme.
RUN sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    printf '\nPort 22\nPermitRootLogin no\n' >> /etc/ssh/sshd_config && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd
EXPOSE 22

RUN wget -q https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && rm /tmp/miniforge.sh
ENV PATH=${CONDA_DIR}/envs/topovelo/bin:${CONDA_DIR}/bin:${PATH}

# Match TopoVelo's Python 3.9/3.10 support and legacy scVelo dependencies.
RUN mamba create -y -n topovelo -c conda-forge python=3.10 pip && \
    mamba clean -a -y

# Keep the same CUDA-capable PyTorch build pattern as the other velocity images.
RUN python -m pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Pin the scientific stack before installing TopoVelo's source code.
RUN python -m pip install --no-cache-dir \
    numpy==1.23.5 pandas==1.5.3 scipy==1.13.1 \
    anndata==0.9.1 matplotlib==3.7.3 numba==0.59.1 \
    scanpy==1.9.8 scvelo==0.2.5 scikit-learn==1.4.2 \
    torch-geometric==2.3.0 igraph==0.11.6 leidenalg==0.10.2 \
    hnswlib==0.8.0 loess==2.1.2 pynndescent==0.5.13 \
    seaborn==0.12.2 tqdm==4.62.3 memory_profiler==0.61.0 \
    ipywidgets jupyter 'setuptools<81' wheel

# Upstream setup.py/setup.cfg disagree on some version limits; preserve pins above.
RUN git clone https://github.com/welch-lab/TopoVelo.git /opt/TopoVelo && \
    cd /opt/TopoVelo && git checkout ${TOPOVELO_REF} && \
    python -m pip install --no-cache-dir --no-deps . && \
    chown -R ${USERNAME}:${USERNAME} /opt/TopoVelo
WORKDIR /opt/TopoVelo

RUN python -c "import torch, scanpy, scvelo, topovelo; print('TopoVelo import OK; PyTorch:', torch.__version__)"

RUN printf '\n# TopoVelo environment\nexport PATH=/opt/conda/envs/topovelo/bin:/opt/conda/bin:$PATH\ncd /opt/TopoVelo\n' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

CMD ["/bin/bash", "-lc", "mkdir -p /run/sshd && /usr/sbin/sshd -D -e"]
