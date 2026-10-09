FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
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

RUN mamba update -n base --all -y

ENV CONDA_ENV=moflow

RUN conda create -n ${CONDA_ENV} -y python=3.7 && \
    /bin/bash -c "source activate ${CONDA_ENV} && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install moflow"

# MoFlow 0.1.1 (the latest release on PyPI) cannot be imported out of the box.
# Three upstream defects have to be fixed here; each fix is verified at build time.
#
# (1) torch 1.13.1 (the newest torch available for python 3.7) lists 'collate' in
#     torch.utils.data.__all__ but never actually imports it, so
#     `from torch.utils.data import *` (MoFlow/moflow.py) raises
#     "AttributeError: module 'torch.utils.data' has no attribute 'collate'".
#     Dropping the spurious entry is the upstream fix (torch >= 1.13.2 fixes it).
# (2) MoFlow/eval_dtw.py has a stray `):` right after the plot_dtw() signature,
#     which is a SyntaxError. MoFlow/__init__.py imports plot_dtw, so
#     `import MoFlow` fails for everyone.
# (3) MoFlow/eval_dtw.py imports fastdtw but the package never declares it.
RUN set -eu; \
    PKG=/opt/conda/envs/${CONDA_ENV}/lib/python3.7/site-packages; \
    TORCH_DATA_INIT="$PKG/torch/utils/data/__init__.py"; \
    sed -i "/^[[:space:]]*'collate',$/d" "$TORCH_DATA_INIT"; \
    ! grep -q "^[[:space:]]*'collate',$" "$TORCH_DATA_INIT"; \
    EVAL_DTW="$PKG/MoFlow/eval_dtw.py"; \
    sed -i '/figsave=None):/{n;/^[[:space:]]*):[[:space:]]*$/d}' "$EVAL_DTW"; \
    /opt/conda/envs/${CONDA_ENV}/bin/python -m py_compile "$EVAL_DTW"; \
    /opt/conda/envs/${CONDA_ENV}/bin/python -m pip install fastdtw; \
    /opt/conda/envs/${CONDA_ENV}/bin/python -c "import MoFlow as mf; print('MoFlow import OK:', mf.__file__, mf.MOFlow)"

RUN echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate moflow to activate the environment."' >> /etc/bash.bashrc

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]
