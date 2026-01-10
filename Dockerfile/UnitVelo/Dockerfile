FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=080528
ARG USER_UID=1000
ARG USER_GID=1000


# Update the system and install required tools
#RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
#    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget\
        openssh-server \
        bzip2 \
        gosu \
        sudo \
        git \
        vim \
        iproute2 \
        htop \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a non-root user
# -m: create home directory
# -s: specify default shell
# -u, -g: specify UID and GID
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}
RUN echo "${USERNAME}:${PASSWORD}" | chpasswd
# Grant the new user sudo privileges (optional but convenient)
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --- SSH configuration ---

# 1. Create necessary directories for the SSH service
RUN mkdir -p /var/run/sshd \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config\
    && echo "Port 22" >> /etc/ssh/sshd_config
RUN sed -i '$ a Port 22' /etc/ssh/sshd_config
COPY start.sh /start.sh
# 2. Make the script executable (very important!)
RUN sed -i 's/\r$//' /start.sh && \
    chmod +x /start.sh

# Expose container port 22
EXPOSE 22

# === ADDED ===
# Create directory for mounting data
RUN mkdir -p /data/simdata

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize Conda so it's available in bash shell
RUN conda init bash

# Create a welcome script that lists all available environments
RUN echo 'display_resources() { \
    echo "CPU threads: $(grep -c '"'"'^processor'"'"' /proc/cpuinfo)"; \
    mem_total_kb=$(grep '"'"'MemTotal'"'"' /proc/meminfo | awk '"'"'{print $2}'"'"'); \
    echo "Total memory: $(echo "$mem_total_kb" | awk '"'"'{printf "%.2f GB", $1/1024/1024}'"'"')"; \
    mem_limit_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); \
    if [ "$mem_limit_bytes" -ge 9223372036854771712 ]; then \
        echo "Memory limit: unlimited"; \
    else \
        echo "Memory limit: $(echo "$mem_limit_bytes" | awk '"'"'{printf "%.2f GB", $1/1024/1024/1024}'"'"')"; \
    fi; \
    cpu_quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); \
    cpu_period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); \
    if [ "$cpu_quota" = -1 ]; then \
        echo "CPU limit: unlimited"; \
    else \
        echo "CPU limit: ${cpu_quota}/${cpu_period} ($(echo "scale=2; $cpu_quota / $cpu_period" | bc) cores)"; \
    fi \
}' >> /etc/bash.bashrc && \
    echo 'echo "Welcome to the single-cell analysis environment!"' >> /etc/bash.bashrc && \
    echo 'echo "Use conda activate <env_name> to activate an environment."' >> /etc/bash.bashrc && \
    echo 'echo "Available environments: UnitVelo"' >> /etc/bash.bashrc && \
    echo 'echo "Data from the host is mounted at /data/simdata"' >> /etc/bash.bashrc && \
    echo 'echo "Container resource information:"' >> /etc/bash.bashrc && \
    echo 'display_resources' >> /etc/bash.bashrc

# 5. UnitVelo
# Accept terms of service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n UnitVelo --yes python=3.9.0 && \
    /bin/bash -c "source $CONDA_DIR/bin/activate unitvelo && pip install unitvelo GPUtil pynvml"
# RUN sed -i -e '1i import scanpy as sc' -e 's/scv.read/sc.read/g' /opt/conda/envs/unitvelo/lib/python3.8/site-packages/unitvelo/utils.py
#     sed -i 's/collections.abc/typing' /opt/conda/envs/unitvelo/lib/python3.8/site-packages/scvelo/logging/_logging.py
#scvelo==0.2.5 tensorflow==2.4.1 numba==0.57.1 numpy==1.21.1

USER root
# Ensure .bashrc exists and belongs to user1
RUN touch /home/${USERNAME}/.bashrc \
 && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc
# Append environment variables so activate/conda are in PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' >> /home/${USERNAME}/.bashrc \
 && echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/${USERNAME}/.bashrc \
 && chown user1:user1 /home/${USERNAME}/.bashrc

# Set the default startup command so the container enters a bash shell on start
# -D: run sshd in the foreground to prevent the container from exiting immediately
CMD ["/start.sh"]

