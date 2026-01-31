FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
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
    echo "${USERNAME}:xxxx" | chpasswd

EXPOSE 2222

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

RUN mamba update -n base --all -y

RUN conda create -n tfvelo -y python=3.8.12 && \
    /bin/bash -c "source activate tfvelo && \
    pip install pandas==1.2.3 && \
    pip install anndata==0.8.0 && \
    pip install scanpy==1.8.2 && \
    pip install numpy==1.21.6 && \
    pip install scipy==1.10.1 && \
    pip install numba==0.57.0 && \
    pip install matplotlib==3.3.4 && \
    pip install scvelo==0.2.4 && \
    pip install typing_extensions"

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate tfvelo to activate the environment."' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
