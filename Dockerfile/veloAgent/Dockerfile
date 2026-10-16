FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

ARG VELOAGENT_REF=main

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

# veloAgent metadata pins Python 3.8 with PyTorch 2.2.2 (CUDA 12.1).
RUN mamba create -y -n veloagent -c conda-forge python=3.8.19 pip && \
    mamba clean -a -y

RUN /bin/bash -c "source activate veloagent && \
    pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121"

# Version pins listed by the upstream environment.yml / pyproject.toml.
RUN /bin/bash -c "source activate veloagent && \
    pip install --no-cache-dir \
    numpy==1.24.4 pandas==2.0.3 anndata==0.9.2 scipy==1.10.1 \
    scikit-learn==1.1.3 seaborn==0.13.2 statsmodels==0.14.1 \
    matplotlib==3.7.3 tqdm==4.66.5 scanpy==1.9.8 scvelo==0.3.2 \
    leidenalg==0.10.2 mesa==2.1.5 mplscience==0.0.7 statannotations==0.7.2 \
    'setuptools>=61' 'pip<25'"

# VeloAE is installed from its upstream repository; veloAgent is installed without
# replacing the pinned dependencies.
RUN /bin/bash -c "source activate veloagent && \
    pip install --no-cache-dir 'git+https://github.com/qiaochen/VeloAE.git'" && \
    git clone https://github.com/mcgilldinglab/veloAgent.git /opt/veloAgent && \
    cd /opt/veloAgent && git checkout ${VELOAGENT_REF} && \
    /bin/bash -c "source activate veloagent && pip install --no-cache-dir --no-deps ." && \
    chown -R ${USERNAME}:${USERNAME} /opt/veloAgent

ENV PYTHONPATH=/opt/veloAgent/src:${PYTHONPATH}

# Verify the package imports during build; GPU detection needs a GPU-enabled runtime.
RUN /bin/bash -c "source activate veloagent && \
    python -c \"import torch, scanpy, scvelo, veloagent; print('veloAgent import OK; PyTorch:', torch.__version__)\""

RUN touch /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /home/${USERNAME}/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/${USERNAME}/.bashrc && \
    echo 'export VELOAGENT_PROTEIN_DIR=/opt/veloAgent' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate veloagent to activate the environment."' >> /etc/bash.bashrc && \
    echo 'echo "Data is mounted from the host at /data/simdata"' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
