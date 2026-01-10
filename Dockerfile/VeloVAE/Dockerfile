# VeloVAE Docker Environment
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git sudo openssh-server vim \
    curl build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev \
    zlib1g-dev libbz2-dev liblzma-dev libreadline-dev libsqlite3-dev \
    libopenblas-dev liblapack-dev libhdf5-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libfontconfig1 python3-dev libcurl4-openssl-dev libatlas-base-dev \
    inkscape && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# SSH configuration (optional - set password if needed)
RUN mkdir -p /var/run/sshd && \
    sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
# Uncomment and modify the next line to set a password:
# RUN echo "${USERNAME}:your_password_here" | chpasswd

EXPOSE 22

# Install Miniforge (latest version)
ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    $CONDA_DIR/bin/conda init bash

ENV PATH=$CONDA_DIR/bin:$PATH

# Configure conda
RUN echo 'auto_activate_base: true' > $CONDA_DIR/.condarc && \
    echo 'channel_priority: strict' >> $CONDA_DIR/.condarc && \
    echo 'show_channel_urls: true' >> $CONDA_DIR/.condarc

# Grant permissions to user
RUN chown -R ${USERNAME}:${USERNAME} $CONDA_DIR

# Update base environment
RUN mamba update --all -y -n base

# Create velovae environment with all dependencies
RUN mamba create -n velovae -y python=3.10 && \
    mamba install -n velovae -c pytorch -c nvidia -c conda-forge -y \
    numpy=1.26.0 scipy=1.12.0 pandas=2.2.3 scikit-learn=1.5.2 \
    scanpy=1.9.8 anndata=0.10.9 scvelo=0.3.3 matplotlib=3.6.3 \
    umap-learn=0.5.7 python-igraph leidenalg \
    pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=11.8 \
    pynvml psutil tqdm fsspec \
    cairosvg reportlab svglib && \
    /bin/bash -c "source $CONDA_DIR/bin/activate velovae && pip install loess"

# Clone VeloVAE to home directory and add to PYTHONPATH
RUN git clone https://github.com/welch-lab/VeloVAE.git /home/${USERNAME}/VeloVAE && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/VeloVAE

# Setup conda activation and environment in user shell
RUN echo 'eval "$(conda shell.bash hook)"' >> /root/.bashrc && \
    echo 'eval "$(conda shell.bash hook)"' >> /home/${USERNAME}/.bashrc && \
    echo 'conda activate velovae' >> /home/${USERNAME}/.bashrc && \
    echo "export PYTHONPATH=\"/home/${USERNAME}/VeloVAE:\$PYTHONPATH\"" >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER ${USERNAME}

# Start SSH service and bash
CMD sudo service ssh start && /bin/bash
