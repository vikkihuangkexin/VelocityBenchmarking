FROM continuumio/miniconda3:latest

ARG USERNAME=user1
ARG PASSWORD=password

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV=sckinetics
ENV PATH=/opt/conda/envs/sckinetics/bin:/opt/conda/bin:$PATH

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo git curl wget ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd && \
    useradd -m -s /bin/bash ${USERNAME} && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

RUN conda create -n ${CONDA_ENV} -y python=3.10 pip && conda clean -afy

RUN /opt/conda/envs/${CONDA_ENV}/bin/python -m pip install \
    numpy==1.26.4 \
    pandas==2.3.3 \
    scipy==1.15.2 \
    matplotlib==3.10.8 \
    scanpy==1.11.5 \
    tqdm==4.67.3 \
    scvelo==0.3.4

RUN /opt/conda/envs/${CONDA_ENV}/bin/python -m pip install \
    MOODS-python \
    biopython \
    ucsc-genomes-downloader \
    gimmemotifs

RUN conda install -n ${CONDA_ENV} -y -c conda-forge r-base rpy2 && \
    conda clean -afy

RUN /opt/conda/envs/${CONDA_ENV}/bin/python -c "import argparse, contextlib, gc, io, logging, os, pickle, re, subprocess, sys, warnings; from datetime import datetime; from pathlib import Path; import matplotlib as mpl; import numpy as np; import pandas as pd; import scanpy as sc; import scipy; from scipy import sparse; import tqdm as tqdm_pkg; from tqdm.auto import tqdm; print('Python:', sys.version); print('NumPy:', np.__version__); print('Pandas:', pd.__version__); print('SciPy:', scipy.__version__); print('Matplotlib:', mpl.__version__); print('Scanpy:', sc.__version__); print('tqdm:', tqdm_pkg.__version__); print('sckinetics environment validation OK')"

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}" >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

WORKDIR /home/${USERNAME}

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
