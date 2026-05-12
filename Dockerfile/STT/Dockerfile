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


RUN echo 'echo "Conda Environment avaliable: stt"' >> /etc/bash.bashrc && \

    echo '' >> /etc/bash.bashrc
RUN echo 'eval "$(mamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'eval "$(mamba shell hook --shell bash)"' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc




RUN mamba create -n stt python==3.9.22 --yes && \
    mamba install petsc --yes && \
    mamba install slepc --yes && \
    mamba install -c conda-forge petsc4py=3.23.0 --yes && \
    mamba install -c conda-forge slepc4py=3.23.0 --yes
RUN /bin/bash -c "source $CONDA_DIR/bin/activate stt && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install scanpy==1.8.0 scvelo==0.3.3 && \
    pip install matplotlib==3.5.3 && \
    pip install plotly==6.0.1 && \
    pip install scipy==1.13.1 scikit-learn==1.6.1 && \
    pip install numpy==1.23.5 pandas==1.2.4"


COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
# CMD ["/bin/bash"]