FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# Use Tsinghua Ubuntu mirrors
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG USER_UID=1000
ARG USER_GID=1000

# Install system packages
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git sudo openssh-server vim \
    tree curl unzip zip build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev libbz2-dev liblzma-dev \
    libreadline-dev libsqlite3-dev libopenblas-dev liblapack-dev \
    libhdf5-dev libncurses5-dev libgdbm-dev libnss3-dev libgl1-mesa-glx \
    libglib2.0-0 libsm6 libxrender1 libxext6 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

# Grant sudo access
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# SSH configuration
RUN mkdir -p /var/run/sshd && \
    sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#PubkeyAuthentication yes/PubkeyAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#AuthorizedKeysFile.*$/#AuthorizedKeysFile/' /etc/ssh/sshd_config && \
    echo "${USERNAME}:xxxx" | chpasswd

EXPOSE 2222

# Data directory
RUN mkdir -p /data/simdata

# Install Miniforge
ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Set permissions
RUN chown -R ${USERNAME}:${USERNAME} /opt/conda

# Initialize conda
RUN $CONDA_DIR/bin/conda init bash

# Configure conda and pip mirrors
RUN echo 'auto_activate_base: true' > $CONDA_DIR/.condarc && \
    echo 'always_copy: true' >> $CONDA_DIR/.condarc && \
    echo 'channel_priority: strict' >> $CONDA_DIR/.condarc && \
    echo 'show_channel_urls: true' >> $CONDA_DIR/.condarc && \
    echo 'notify_outdated_conda: false' >> $CONDA_DIR/.condarc && \
    echo 'default_channels: []' >> $CONDA_DIR/.condarc && \
    echo 'channels:' >> $CONDA_DIR/.condarc && \
    echo '  - conda-forge' >> $CONDA_DIR/.condarc && \
    echo 'custom_channels:' >> $CONDA_DIR/.condarc && \
    echo '  conda-forge: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/' >> $CONDA_DIR/.condarc && \
    echo '  bioconda: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/' >> $CONDA_DIR/.condarc && \
    echo '  pytorch: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/' >> $CONDA_DIR/.condarc && \
    echo '  auto: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/' >> $CONDA_DIR/.condarc && \
    mkdir -p /root/.pip && \
    echo '[global]' > /root/.pip/pip.conf && \
    echo 'index-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple' >> /root/.pip/pip.conf && \
    echo 'trusted-host = mirrors.tuna.tsinghua.edu.cn' >> /root/.pip/pip.conf && \
    mkdir -p /home/${USERNAME}/.pip && \
    cp /root/.pip/pip.conf /home/${USERNAME}/.pip/ && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.pip

# Update base environment
RUN mamba update -n base --all -y

# Create deepvelo environment
RUN mamba create -n deepvelo -y python=3.9 && \
    /bin/bash -c "source activate deepvelo && pip install deepvelo"

# Welcome message
RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate deepvelo to activate the environment."' >> /etc/bash.bashrc

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
