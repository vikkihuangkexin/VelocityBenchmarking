FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# ============================================================
# CRAK-Velo Dockerfile
# GitHub: https://github.com/StatBiomed/CRAK-Velo
# Python 3.8 | scanpy 1.8.2 | scVelo 0.2.5 | unitvelo | pybedtools
# ============================================================

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV=crakvelo

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        git \
        sudo \
        vim \
        openssh-server \
        build-essential \
        libhdf5-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /var/run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#PubkeyAuthentication yes/PubkeyAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#AuthorizedKeysFile.*$/#AuthorizedKeysFile/' /etc/ssh/sshd_config && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd

EXPOSE 22

RUN mkdir -p /data/simdata

ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN chown -R ${USERNAME}:${USERNAME} /opt/conda

RUN $CONDA_DIR/bin/conda init bash

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

# ============================================================
# CRAK-Velo environment
# ============================================================
RUN conda create -n ${CONDA_ENV} --yes python=3.8 pip && \
    conda clean -afy

RUN /bin/bash -c "source activate ${CONDA_ENV} && \
    pip install scanpy==1.8.2 scvelo==0.2.5 anndata==0.8.0 umap-learn==0.5.3 \
        numba==0.55.2 llvmlite==0.38.1 numpy==1.21.6 scipy==1.7.3 && \
    pip install unitvelo && \
    pip install matplotlib==3.5.3 pandas==1.5.3"

RUN conda install -n ${CONDA_ENV} --yes --channel conda-forge --channel bioconda pybedtools bedtools && \
    conda clean -afy

RUN /bin/bash -c "source activate ${CONDA_ENV} && \
    pip install --no-cache-dir --force-reinstall --no-deps numpy==1.22.4"

# Verify the environment during the build
RUN /opt/conda/envs/${CONDA_ENV}/bin/python -c "import numpy, scipy, anndata, scanpy, scvelo, numba, llvmlite, pandas, matplotlib, pybedtools; print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('Scanpy:', scanpy.__version__); print('scVelo:', scvelo.__version__); print('Numba:', numba.__version__); print('Pandas:', pandas.__version__)"

# ============================================================
# CRAK-Velo sources (upstream clone; the project is a script repository,
# there is no installable Python package)
# ============================================================
ARG CRAK_VELO_REPO=https://github.com/StatBiomed/CRAK-Velo.git

RUN git clone --depth 1 ${CRAK_VELO_REPO} /opt/CRAK-Velo && \
    test -f /opt/CRAK-Velo/crak-velo/main.py

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate crakvelo to activate the environment."' >> /etc/bash.bashrc && \
    echo 'echo "Available environment: crakvelo"' >> /etc/bash.bashrc && \
    echo 'echo "Data has been mounted from the host to /data/simdata"' >> /etc/bash.bashrc

ENV PATH=$CONDA_DIR/envs/crakvelo/bin:$CONDA_DIR/bin:$PATH
ENV CONDA_DEFAULT_ENV=crakvelo

COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && \
    chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
