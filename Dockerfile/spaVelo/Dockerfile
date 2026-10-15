FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

ARG SPAVELO_REF=main

RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git sudo openssh-server vim \
    tree curl unzip zip build-essential libgomp1 libgl1-mesa-glx libglib2.0-0 && \
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

# spaVelo environment: Python 3.11 with PyTorch 2.2.1 (CUDA 12.1).
RUN mamba create -y -n spavelo -c conda-forge python=3.11.8 pip=24.0 && \
    mamba install -y -n spavelo -c pytorch -c nvidia -c conda-forge \
    pytorch=2.2.1 pytorch-cuda=12.1 && \
    mamba clean -a -y

# Pinned scientific stack aligned with the upstream environment.yml.
RUN /bin/bash -c "source activate spavelo && \
    pip install --no-cache-dir \
    'numpy==1.26.4' 'pandas==2.2.1' 'scipy==1.12.0' \
    'anndata==0.10.6' 'scanpy==1.9.8' 'scvelo==0.3.2' \
    'scikit-learn==1.4.2' 'matplotlib==3.8.3' 'seaborn==0.13.2' \
    'h5py==3.10.0' 'numba==0.59.1' 'leidenalg==0.10.2' \
    'pyro-ppl==1.9.0' 'jax==0.4.20' 'jaxlib==0.4.20' \
    'ipykernel==6.28.0' 'tqdm' 'loompy==3.0.7'"

# The upstream repository is script-based (no setup.py), so it is cloned and used
# through the wrapper's --repo-dir flag.
RUN git clone https://github.com/1062638515/spavelo.git /opt/spavelo && \
    cd /opt/spavelo && git checkout ${SPAVELO_REF} && \
    chown -R ${USERNAME}:${USERNAME} /opt/spavelo

RUN /bin/bash -c "source activate spavelo && \
    python -c \"import torch, scanpy, scvelo; print('spaVelo stack import OK; PyTorch:', torch.__version__)\""

RUN touch /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /home/${USERNAME}/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/${USERNAME}/.bashrc && \
    echo 'export SPAVELO_REPO_DIR=/opt/spavelo' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate spavelo to activate the environment."' >> /etc/bash.bashrc && \
    echo 'echo "Data is mounted from the host at /data/simdata"' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
