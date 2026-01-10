### Overview
This repository provides Dockerfiles to build tool-specific images for single‑cell analysis workflows. Each image includes an SSH server and a non‑root user to simplify remote access and reproducible execution.

---

## Build an image for a specific tool
To build an image for a tool, change into the tool’s directory with the key file `Dockerfile` and run:

```bash
docker build -t TOOL-name:tag .
```

This will produce a Docker image named `TOOL-name` with the specified `tag`.

---

### SSH user and customization
All images include an external SSH service and a non‑root user. You can customize the default user and credentials by editing these build arguments in the Dockerfile:

```dockerfile
ARG USERNAME=user1
ARG PASSWORD=user1
ARG USER_UID=1000
ARG USER_GID=1000
```

Change the values to match your preferred username, password, UID and GID before building.

---

### Conda environments and package versions
Each tool is installed into its own Conda environment inside the image. If you need to change package versions or add dependencies, edit the corresponding `RUN conda create` and `pip install` lines in the Dockerfile.

Example for creating the `unitvelo` environment and installing packages:

```dockerfile
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n unitvelo --yes python=3.9.0 && \
    /bin/bash -c "source $CONDA_DIR/bin/activate unitvelo && \
    pip install unitvelo GPUtil pynvml scvelo==0.2.5 tensorflow==2.4.1 numba==0.57.1 numpy==1.21.1"
```

Edit the package list and versions to suit your requirements.

---

## Run Container

### Interactive Mode

```bash
docker run --gpus all -it --rm \
  -v /path/to/data:/workspace/data \
  TOOL-name:tag
```

### With SSH Access

```bash
docker run --gpus all -d \
  -p 2222:22 \
  -v /path/to/data:/workspace/data \
  --name TOOL-name-container \
  TOOL-name:tag
```

Connect via SSH:
```bash
ssh UserNameYouSet@localhost -p 2222
```

### Important notes about entrypoint and startup scripts
- **docker-entrypoint.sh**  
  This script is used to mount local paths into the container at runtime. If your environment does not support this behavior, comment out or remove the related lines in the Dockerfile:

  ```dockerfile
  COPY docker-entrypoint.sh /usr/local/bin/
  RUN chmod +x /usr/local/bin/docker-entrypoint.sh
  ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
  ```

- **start.sh**  
  Use `start.sh` for container startup tasks such as starting the SSH daemon or applying runtime source code modifications. We recommend placing any runtime operations you need inside this script so they run automatically when the container starts.

---

### Quick tips
- Keep environment changes minimal in the Dockerfile to preserve reproducibility.
- If you change user credentials or UIDs, make sure mounted volumes have compatible ownership or adjust permissions at runtime.
- Test builds locally before deploying to shared environments.
