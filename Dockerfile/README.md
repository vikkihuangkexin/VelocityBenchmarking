# Docker environments
This repository provides Dockerfiles to build tool-specific images for single-cell analysis workflows. Each image includes an SSH server and a non-root user to simplify remote access and reproducible execution.

---

## Build an image for a specific tool
To build an image for a tool, change into its directory containing `Dockerfile` and `start.sh`, then run:

```bash
docker build -t tool-name:tag .
```

This produces an image named `tool-name` with the specified `tag`.

### DeepVelo Python 3.11 recipe

This recipe installs the [team fork](https://github.com/sd68515/DeepVelo_py311)
using `environment-py311.yml`, then installs CPU PyTorch 2.5.1 and the fork
in editable mode. The environment is `deepvelo_py311`, available on `PATH`.
From the repository root:

```bash
docker build -t deepvelo:revision2 Dockerfile/DeepVelo
docker run --rm -v "$PWD:/workspace" -w /workspace deepvelo:revision2 \
  python Benchmarked_tools/DeepVelo/DeepVelo.py \
  --input data/sample.h5ad --output-dir results/deepvelo \
  --kind real --coordinate-key X_umap --cluster-key cell_type
```

Inputs must meet the [DeepVelo contract](../Benchmarked_tools/DeepVelo/README.md).
Results under `results/deepvelo` are written to the mounted checkout. Set
`--build-arg DEEPVELO_REF=<commit-or-tag>` to select a fork revision;
the default is `main`. CUDA use needs a compatible PyTorch installation and
container runtime configuration in addition to GPU access.

---

### SSH user and customization
The recipes include an SSH service and a non-root user. Check the selected
Dockerfile for supported build arguments; DeepVelo accepts:

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
  tool-name:tag
```

### With SSH Access

```bash
docker run --gpus all -dit \
  -p 127.0.0.1:2222:22 \
  -v /path/to/data:/workspace/data \
  --name tool-name-container \
  tool-name:tag
```

Connect via SSH:
```bash
ssh UserNameYouSet@localhost -p 2222
```

### Important notes about entrypoint and startup scripts
- **docker-entrypoint.sh**  
  Some recipes use this script for extra startup setup. Inspect the selected
  script before changing its `ENTRYPOINT`; host directories are mounted using
  `docker run -v`, as in the examples above:

  ```dockerfile
  COPY docker-entrypoint.sh /usr/local/bin/
  RUN chmod +x /usr/local/bin/docker-entrypoint.sh
  ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
  ```

- **start.sh**  
  DeepVelo's `start.sh` starts SSH and opens Bash. An explicit command after
  the image name replaces the default `CMD`. Other recipes may use different
  entrypoints or environment names.

---

### Quick tips
- Keep environment changes minimal in the Dockerfile to preserve reproducibility.
- If you change user credentials or UIDs, make sure mounted volumes have compatible ownership or adjust permissions at runtime.
- Test builds locally before deploying to shared environments.
