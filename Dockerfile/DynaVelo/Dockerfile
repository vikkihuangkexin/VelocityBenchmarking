# DynaVelo - latent neural ODE model that learns multiomic (RNA expression + TF motif
# accessibility) velocities, infers dynamic gene regulations, and performs in-silico
# perturbations.
# Upstream repository: https://github.com/karbalayghareh/DynaVelo
# Base image: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Environment: conda env `dynavelo` (python 3.9, torch 1.12.1 CUDA, scvelo 0.2.4).
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

# System dependencies required by the method.
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
# dynavelo environment (python 3.9). The version pins mirror the upstream
# environment.yml. That snapshot locks several exact conda builds that were later
# removed from the official channels, so the same version numbers are reproduced
# here with "create an empty conda env + pip install" instead of
# `conda env create -f environment.yml`.
# ---------------------------------------------------------------------------
RUN conda create -n dynavelo --yes python=3.9
ENV DYNAVELO_ENV=/opt/conda/envs/dynavelo

# Toolchain pins: pip<24.1 because torchsde==0.2.5 ships invalid metadata that newer
# pip rejects, and setuptools<81 because numba==0.53.1 / llvmlite==0.36.0 still import
# pkg_resources, which setuptools>=81 removed.
RUN /opt/conda/envs/dynavelo/bin/python -m pip install --no-cache-dir -U wheel "pip<24.1" "setuptools<81"

# All pins are resolved in a single pip command on purpose: installing them one by one
# lets pip reset already pinned versions while resolving later packages.
RUN /opt/conda/envs/dynavelo/bin/python -m pip install --no-cache-dir \
        torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
        numpy==1.22.4 scipy==1.10.1 pandas==1.5.3 scikit-learn==1.0.2 \
        matplotlib==3.7.1 h5py==3.7.0 llvmlite==0.36.0 numba==0.53.1 \
        python-igraph==0.10.4 leidenalg==0.9.1 umap-learn==0.5.4 \
        tqdm==4.64.0 psutil==5.9.1 \
        anndata==0.10.2 scanpy==1.9.5 scvelo==0.2.4 \
        torchdiffeq==0.2.3 torchsde==0.2.5 torchmetrics==0.11.1 functorch==0.2.1 \
        multivelo==0.1.2 \
        adjustText==1.2.0 annoy==1.17.3 array-api-compat==1.4 bbknn==1.6.0 boltons==21.0.0 \
        click==8.1.3 Cython==3.0.10 docrep==0.3.2 exceptiongroup==1.1.3 future==0.18.3 \
        harmonypy==0.0.9 jinja2==3.0.3 jsonpickle==3.0.1 loompy==3.0.7 \
        matplotlib-venn==0.11.9 networkx==3.2 numpy-groupies==0.9.19 progressbar2==4.2.0 \
        pygam==0.8.0 pygpcca==1.0.4 python-utils==3.7.0 pyvis==0.3.2 \
        scikit-misc==0.3.0 session-info==1.0.0 sparse==0.14.0 stdlib-list==0.9.0 \
        tbb==2021.10.0 trampoline==0.1.2 utils==1.0.2 wrapt==1.15.0 \
        nvidia-ml-py GPUtil ipywidgets

# Version assertions: fail the build if any later pip layer silently reset a pin.
RUN /opt/conda/envs/dynavelo/bin/python -c "import numpy, scipy, pandas, matplotlib, sklearn, h5py, numba, llvmlite, anndata, scanpy, scvelo, torch, torchdiffeq, torchsde, torchmetrics, functorch, multivelo, psutil, pynvml, GPUtil, tqdm, ipywidgets; import umap, networkx; assert numpy.__version__ == '1.22.4', 'numpy was reset to ' + numpy.__version__; assert scipy.__version__ == '1.10.1'; assert pandas.__version__ == '1.5.3'; assert matplotlib.__version__ == '3.7.1'; assert sklearn.__version__ == '1.0.2', 'scikit-learn was reset to ' + sklearn.__version__; assert h5py.__version__ == '3.7.0'; assert numba.__version__ == '0.53.1'; assert llvmlite.__version__ == '0.36.0'; assert networkx.__version__ == '3.2'; assert tqdm.__version__ == '4.64.0'; assert umap.__version__ == '0.5.4', 'umap-learn was reset to ' + umap.__version__; assert torch.__version__.startswith('1.12.1'), 'torch was reset to ' + torch.__version__; assert torch.version.cuda is not None, 'torch has no CUDA runtime (CPU-only wheel)'; print('dynavelo env import + version check OK -> numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, '| sklearn', sklearn.__version__, '| matplotlib', matplotlib.__version__, '| h5py', h5py.__version__, '| numba', numba.__version__, '| anndata', anndata.__version__, '| scanpy', scanpy.__version__, '| scvelo', scvelo.__version__, '| umap', umap.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---------------------------------------------------------------------------
# Upstream source. The source is cloned from GitHub; the previous local vendored
# COPY of the source and the pre-downloaded wheels/ offline install are not used.
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/karbalayghareh/DynaVelo.git /opt/DynaVelo
ENV PYTHONPATH=/opt/DynaVelo
RUN /opt/conda/envs/dynavelo/bin/python -c "import dynavelo.models as m; from dynavelo.models import MultiomeDataset, DynaVelo; from dynavelo.utils import Logger; print('DynaVelo source import check OK ->', m.__file__)"

# Make any user / non-interactive shell resolve the dynavelo environment and the source.
RUN chown -R ${USERNAME}:${USERNAME} /opt/conda/envs/dynavelo && \
    printf '%s\n' \
      '. /opt/conda/etc/profile.d/conda.sh' \
      'conda activate dynavelo 2>/dev/null || true' \
      'export PYTHONPATH=/opt/DynaVelo:$PYTHONPATH' \
      >> /etc/bash.bashrc && \
    printf '%s\n' \
      '. /opt/conda/etc/profile.d/conda.sh' \
      'conda activate dynavelo 2>/dev/null || true' \
      'export PYTHONPATH=/opt/DynaVelo:$PYTHONPATH' \
      >> /root/.bashrc
ENV PATH=/opt/conda/envs/dynavelo/bin:$PATH

RUN echo 'echo "Welcome to the single-cell multiomics analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate dynavelo to activate the environment."' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
