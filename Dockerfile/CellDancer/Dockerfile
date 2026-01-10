# Use NVIDIA's official CUDA image as the base to ensure GPU driver compatibility
# You can choose a more suitable CUDA version based on your NVIDIA driver
# e.g., nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000


# Update the system and install required tools
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git
RUN apt-get install sudo
RUN apt-get -y install openssh-server gosu
RUN apt-get -y install vim
RUN apt-get install -y iproute2
RUN apt-get install -y htop
RUN rm -rf /var/lib/apt/lists/*

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

COPY start.sh /start.sh

# 2. Make the script executable (very important!)
RUN chmod +x /start.sh

# Expose container port 22
EXPOSE 22

# === ADDED ===
# Create directory for mounting data
RUN mkdir -p /data/simdata

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

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
    echo 'echo "Available environments: CellDancer"' >> /etc/bash.bashrc && \
    echo 'echo "Data from the host is mounted at /data/simdata"' >> /etc/bash.bashrc && \
    echo 'echo "Container resource information:"' >> /etc/bash.bashrc && \
    echo 'display_resources' >> /etc/bash.bashrc

# =========================================================================
# Create separate environments for tools
# =========================================================================

# CellDancer
RUN conda create -n CellDancer -c conda-forge --yes python=3.7.6 && \
    /bin/bash -c "source $CONDA_DIR/bin/activate CD && pip install celldancer GPUtil scvelo==0.2.5"

USER root
# Ensure .bashrc exists and belongs to the specified user
RUN touch /home/${USERNAME}/.bashrc \
 && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc
# Append environment variables so activate/conda are in PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' >> /home/${USERNAME}/.bashrc \
 && echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/${USERNAME}/.bashrc \
 && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc


# Set the default startup command so the container enters a bash shell on start
# -D: run sshd in the foreground to prevent the container from exiting immediately
CMD ["/start.sh"]

