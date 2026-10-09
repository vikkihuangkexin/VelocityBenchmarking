FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git sudo openssh-server vim \
    tree curl unzip zip build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev libbz2-dev liblzma-dev \
    libreadline-dev libsqlite3-dev libopenblas-dev liblapack-dev \
    libhdf5-dev libncurses5-dev libgdbm-dev libnss3-dev libgl1-mesa-glx \
    libglib2.0-0 libsm6 libxrender1 libxext6 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /var/run/sshd && \
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

# SvelvetVAE (velvetVAE) environment: Python 3.9, PyTorch 2.0.1, JAX 0.4.13, scvi-tools 0.19.0.
RUN conda create -n svelvetvae --yes python=3.9.0

RUN /opt/conda/envs/svelvetvae/bin/pip install 'pip<24.1'
RUN /opt/conda/envs/svelvetvae/bin/pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN /opt/conda/envs/svelvetvae/bin/pip install numpyro==0.12.1 optax==0.1.7
RUN /opt/conda/envs/svelvetvae/bin/pip install scvi-tools==0.19.0 scanpy==1.9.8 scvelo==0.2.5 anndata==0.8.0
RUN /opt/conda/envs/svelvetvae/bin/pip install mudata==0.2.0
RUN /opt/conda/envs/svelvetvae/bin/pip install numba==0.58.1 llvmlite==0.41.1 scipy==1.8.1
RUN /opt/conda/envs/svelvetvae/bin/pip install pandas==1.5.3 scikit-learn scikit-misc==0.2.0
RUN /opt/conda/envs/svelvetvae/bin/pip install torchmetrics==0.11.4 torchsde leidenalg
RUN /opt/conda/envs/svelvetvae/bin/pip install matplotlib==3.5.2 seaborn plotly==5.10.0 adjustText tqdm
RUN /opt/conda/envs/svelvetvae/bin/pip install tslearn==0.5.3 pymde ipykernel 'setuptools<81' numpy==1.23.5
RUN /opt/conda/envs/svelvetvae/bin/pip install 'jax[cuda11_local]==0.4.13' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN /opt/conda/envs/svelvetvae/bin/pip install chex==0.1.7 flax==0.6.11

# Install velvetVAE from the upstream repository.
RUN git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000 && \
    git clone https://github.com/rorymaizels/velvetVAE.git /opt/velvetVAE && \
    /opt/conda/envs/svelvetvae/bin/pip install --no-deps -e /opt/velvetVAE

RUN touch /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /home/${USERNAME}/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate svelvetvae to activate the environment."' >> /etc/bash.bashrc && \
    echo 'echo "Data is mounted from the host at /data/simdata"' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
