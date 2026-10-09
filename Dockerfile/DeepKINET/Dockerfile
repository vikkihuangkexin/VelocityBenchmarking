FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# ============================================================
# DeepKINET Dockerfile
# GitHub: https://github.com/3254c/DeepKINET
# Python 3.9 | PyTorch 2.1.0 CUDA 11.8 | scVelo 0.2.5
# ============================================================

ENV DEBIAN_FRONTEND=noninteractive

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
# DeepKINET environment
# ============================================================
# Note: the upstream image annotated a Zenodo snapshot as the only working copy.
# Following the integration conventions, the public GitHub repository is cloned
# and installed in editable mode, which is the maintained source of the package.
RUN conda create -n deepkinet --yes python=3.9.0 && \
    /bin/bash -c "source activate deepkinet && \
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install anndata==0.10.3 && \
    pip install scanpy==1.9.6 && \
    pip install scvelo==0.2.5 && \
    pip install pandas==1.5.3 && \
    pip install numpy==1.23.5 && \
    pip install einops==0.7.0 && \
    pip install leidenalg==0.10.1 && \
    pip install umap-learn==0.5.5 && \
    pip install matplotlib==3.7.1 && \
    pip install seaborn==0.12.2 && \
    git clone https://github.com/3254c/DeepKINET.git /opt/DeepKINET && \
    cd /opt/DeepKINET && \
    pip install -e ."

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate deepkinet to activate the environment."' >> /etc/bash.bashrc && \
    echo 'echo "Available environment: deepkinet"' >> /etc/bash.bashrc && \
    echo 'echo "Data has been mounted from the host to /data/simdata"' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && \
    chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
