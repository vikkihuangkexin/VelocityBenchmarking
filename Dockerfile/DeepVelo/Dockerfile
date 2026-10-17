FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bzip2 ca-certificates build-essential git wget sudo openssh-server \
    libgomp1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /var/run/sshd /data/simdata && \
    printf '%s\n' 'PasswordAuthentication yes' 'PermitRootLogin no' >> /etc/ssh/sshd_config

EXPOSE 22

RUN wget -q https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

ARG DEEPVELO_REF=main
RUN git clone https://github.com/sd68515/DeepVelo_py311.git /opt/DeepVelo_py311 && \
    cd /opt/DeepVelo_py311 && git checkout ${DEEPVELO_REF} && \
    conda env create -f environment-py311.yml && \
    conda run -n deepvelo_py311 python -m pip install \
      --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 && \
    conda run -n deepvelo_py311 python -m pip install --no-deps -e . && \
    chown -R ${USERNAME}:${USERNAME} ${CONDA_DIR} /opt/DeepVelo_py311

ENV PATH=${CONDA_DIR}/envs/deepvelo_py311/bin:${PATH}

COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
