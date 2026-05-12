FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
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
    echo "${USERNAME}:password" | chpasswd

EXPOSE 8848


RUN mkdir -p /data/simdata


ENV CONDA_DIR=/opt/conda

RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH


RUN $CONDA_DIR/bin/conda init bash


RUN echo 'echo "Conda Environment avaliable: pyro"' >> /etc/bash.bashrc && \
    echo '' >> /etc/bash.bashrc
RUN echo 'eval "$(mamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'eval "$(mamba shell hook --shell bash)"' >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

COPY pyrovelocity /tmp/pyrovelocity
WORKDIR /tmp/pyrovelocity
COPY torch-2.6.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl /tmp/
COPY nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl /tmp/
RUN mamba create -n pyro --yes python=3.11
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && mamba install -c conda-forge --yes rust"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && mamba install --yes daft"
RUN /bin/bash -c "source activate pyro && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"

# Install torch and other required packages
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install /tmp/torch-2.6.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install maturin"

# Install pyrovelocity and other Python packages
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install pyrovelocity==0.2.0"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install beartype==0.14.0"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install scvi-tools==1.1.1 --no-deps"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install jax==0.5.3 jaxlib==0.5.3"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install scanpy==1.11.1 scvelo==0.3.3 anndata==0.10.5"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install numpy==1.26.4 pandas==2.2.3"
RUN /bin/bash -c "source $CONDA_DIR/bin/activate pyro && pip install flytekit dataclasses_json hydra_zen dulwich"

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
# CMD ["/bin/bash"]