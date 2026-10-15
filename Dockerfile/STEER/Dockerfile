FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

ARG STEER_REF=master

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

# STEER environment: Python 3.10 with R/mclust for the clustering-based priors.
RUN mamba create -y -n steer -c conda-forge \
    python=3.10 pip r-base=4.3.3 r-mclust=6.1.1 && \
    mamba clean -a -y

# Official CUDA 12.4 PyTorch and PyTorch Geometric stack.
RUN /bin/bash -c "source activate steer && \
    pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html && \
    pip install --no-cache-dir torch-geometric"

# Install the upstream pinned requirements before the package itself.
RUN git clone https://github.com/lzygenomics/STEER.git /opt/STEER && \
    cd /opt/STEER && git checkout ${STEER_REF} && \
    /bin/bash -c "source activate steer && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps ." && \
    chown -R ${USERNAME}:${USERNAME} /opt/STEER

ENV PYTHONPATH=/opt/STEER:${PYTHONPATH}

# Build-time import smoke test (GPU availability is checked at run time).
RUN /bin/bash -c "source activate steer && \
    python -c \"import torch, scanpy, scvelo, torch_geometric, steer; print('STEER import OK; PyTorch:', torch.__version__)\""

RUN touch /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /home/${USERNAME}/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate steer to activate the environment."' >> /etc/bash.bashrc && \
    echo 'echo "Data is mounted from the host at /data/simdata"' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
