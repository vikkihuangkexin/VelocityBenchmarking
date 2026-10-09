# MultiVeloVAE - probabilistic framework for velocity inference from multi-lineage,
# multi-omic and multi-sample single-cell data.
# Upstream repository: https://github.com/welch-lab/MultiVeloVAE
# Base image: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Environment: conda env `MultiVeloVAE` (python 3.10.8, torch 2.0.0, scvelo 0.2.5).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Force the non-interactive matplotlib backend inside a headless container and flush
# Python output immediately so that logs are written in real time.
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

# System dependencies. build-essential / gfortran are required because some
# dependencies (for example hnswlib) contain C++ extensions that pip would otherwise
# try to build from source and fail without a compiler.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        gfortran \
        openssh-server \
        bzip2 \
        gosu \
        sudo \
        git \
        vim \
        iproute2 \
        htop \
        bc \
        procps \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Non-root user.
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}
RUN echo "${USERNAME}:${PASSWORD}" | chpasswd
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# SSH service.
RUN mkdir -p /var/run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "Port 22" >> /etc/ssh/sshd_config

EXPOSE 22

# Uniform data mount point shared by every benchmark image.
RUN mkdir -p /data/simdata

# Miniforge3 bootstrap.
ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN chown -R ${USERNAME}:${USERNAME} /opt/conda

RUN $CONDA_DIR/bin/conda init bash

# Project-standard conda channels and pip index.
RUN echo 'auto_activate_base: true' > $CONDA_DIR/.condarc && \
    echo 'always_copy: true' >> $CONDA_DIR/.condarc && \
    echo 'channel_priority: strict' >> $CONDA_DIR/.condarc && \
    echo 'show_channel_urls: true' >> $CONDA_DIR/.condarc && \
    echo 'notify_outdated_conda: false' >> $CONDA_DIR/.condarc && \
    echo 'default_channels: []' >> $CONDA_DIR/.condarc && \
    echo 'channels:' >> $CONDA_DIR/.condarc && \
    echo '  - conda-forge' >> $CONDA_DIR/.condarc && \
    echo 'custom_channels:' >> $CONDA_DIR/.condarc && \
    echo '  conda-forge: https://conda.anaconda.org/conda-forge' >> $CONDA_DIR/.condarc && \
    echo '  bioconda: https://conda.anaconda.org/bioconda' >> $CONDA_DIR/.condarc && \
    echo '  pytorch: https://conda.anaconda.org/pytorch' >> $CONDA_DIR/.condarc && \
    echo '  auto: https://conda.anaconda.org/auto' >> $CONDA_DIR/.condarc && \
    mkdir -p /root/.pip && \
    echo '[global]' > /root/.pip/pip.conf && \
    echo 'index-url = https://pypi.org/simple' >> /root/.pip/pip.conf && \
    mkdir -p /home/${USERNAME}/.pip && \
    cp /root/.pip/pip.conf /home/${USERNAME}/.pip/ && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.pip

RUN mamba update -n base --all -y

# ---------------------------------------------------------------------------
# MultiVeloVAE environment (python 3.10.8). The version pins match the upstream
# paper-notebooks/reproducible_package_versions.txt used to generate the manuscript
# figures. The upstream export locks several exact conda builds that were later
# removed from the official channels, so the same version numbers are reproduced here
# with "create an empty conda env + pip install" instead of
# `conda env create -f environment.yml`.
# ---------------------------------------------------------------------------
RUN conda create -n MultiVeloVAE --yes python=3.10.8
ENV MULTIVELOVAE_ENV=/opt/conda/envs/MultiVeloVAE

# Toolchain pins: setuptools<81 because numba / llvmlite still import pkg_resources,
# and pip<24.1 for compatibility with the pinned dependency metadata.
RUN /opt/conda/envs/MultiVeloVAE/bin/python -m pip install --no-cache-dir -U wheel "setuptools<81" "pip<24.1"

# All pins are resolved in a single pip command on purpose: installing them one by one
# lets pip reset already pinned versions while resolving later packages.
# On PyPI the `torch==2.0.0` Linux wheel ships the CUDA runtime (cu117), which is
# compatible with this CUDA 11.8 base image.
RUN /opt/conda/envs/MultiVeloVAE/bin/python -m pip install --no-cache-dir \
        multivelovae==0.1.0 \
        torch==2.0.0 \
        numpy==1.23.5 scipy==1.10.1 pandas==1.5.3 scikit-learn==1.2.2 \
        anndata==0.8.0 scanpy==1.9.3 scvelo==0.2.5 \
        matplotlib==3.7.3 seaborn==0.12.2 umap-learn==0.5.3 \
        multivelo==0.1.2 \
        hnswlib igraph leidenalg loess tqdm ipywidgets \
        psutil nvidia-ml-py GPUtil

# Import + version assertions: fail the build if a pin was silently reset or if torch
# was installed as a CPU-only build.
RUN /opt/conda/envs/MultiVeloVAE/bin/python -c "\
import sys, importlib.metadata as md; \
import numpy, scipy, pandas, sklearn, anndata, scanpy, scvelo, matplotlib, umap, tqdm, ipywidgets, psutil, pynvml, GPUtil; \
import torch, multivelo, multivelovae as vv; \
assert hasattr(vv, 'VAEChrom'), 'multivelovae does not expose VAEChrom'; \
want = {'numpy': '1.23.5', 'scipy': '1.10.1', 'pandas': '1.5.3', 'scikit-learn': '1.2.2', 'anndata': '0.8.0', 'scanpy': '1.9.3', 'scvelo': '0.2.5', 'matplotlib': '3.7.3', 'umap-learn': '0.5.3', 'multivelo': '0.1.2', 'multivelovae': '0.1.0'}; \
bad = {k: md.version(k) for k, v in want.items() if md.version(k) != v}; \
assert not bad, 'pip reset these versions: ' + str(bad); \
assert md.version('torch').startswith('2.0.0'), 'torch was reset to ' + md.version('torch'); \
assert torch.version.cuda is not None, 'torch is a CPU build (version.cuda=None)'; \
print('MultiVeloVAE import + version check OK -> python', sys.version.split()[0], '| torch', torch.__version__, 'cuda', torch.version.cuda, '| pivots', {k: md.version(k) for k in want})"

# Make any user / non-interactive shell resolve the MultiVeloVAE environment.
RUN chown -R ${USERNAME}:${USERNAME} /opt/conda/envs/MultiVeloVAE && \
    printf '%s\n' \
      '. /opt/conda/etc/profile.d/conda.sh' \
      'conda activate MultiVeloVAE 2>/dev/null || true' \
      >> /etc/bash.bashrc && \
    printf '%s\n' \
      '. /opt/conda/etc/profile.d/conda.sh' \
      'conda activate MultiVeloVAE 2>/dev/null || true' \
      >> /root/.bashrc
ENV PATH=/opt/conda/envs/MultiVeloVAE/bin:$PATH

RUN echo 'echo "Welcome to the single-cell multiomics analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate MultiVeloVAE to activate the environment."' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
