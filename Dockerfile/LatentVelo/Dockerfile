FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000



RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git && \
    apt-get install -y sudo && \
    apt-get install -y openssh-server vim && \
    rm -rf /var/lib/apt/lists/*


RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /home/${USERNAME}/.ssh && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.ssh && \
    chmod 700 /home/${USERNAME}/.ssh

RUN mkdir -p /var/run/sshd && \
    sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#PubkeyAuthentication yes/PubkeyAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#AuthorizedKeysFile.*$/#AuthorizedKeysFile/' /etc/ssh/sshd_config && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd

EXPOSE 8848


RUN mkdir -p /data/simdata

ENV CONDA_DIR=/opt/conda

RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN $CONDA_DIR/bin/conda init bash


RUN echo 'eval "$(mamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'eval "$(mamba shell hook --shell bash)"' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc


RUN mamba create -n LatentVelo --yes python=3.9
RUN /bin/bash -c "source $CONDA_DIR/bin/activate LatentVelo && pip install scanpy==1.10.3 scvelo==0.3.3 && \
    pip install numpy==1.23.5 pandas==2.2.3"
COPY torch-2.6.0-cp39-cp39-manylinux1_x86_64.whl /tmp/
RUN /bin/bash -c "source $CONDA_DIR/bin/activate LatentVelo && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install /tmp/torch-2.6.0-cp39-cp39-manylinux1_x86_64.whl"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate LatentVelo && \
    git clone https://github.com/Spencerfar/LatentVelo.git /data/simdata/latentvelo && \
    cd /data/simdata/latentvelo && \
    python setup.py install"



COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
# CMD ["/bin/bash"]