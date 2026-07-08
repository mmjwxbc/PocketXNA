ARG CUDA_IMAGE=hub.rat.dev/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=pxm_cu128
ARG MINIFORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    FONTCONFIG_PATH=/etc/fonts

SHELL ["/bin/bash", "-lc"]

# Optional: rewrite apt sources to a faster mirror. Defaults to Aliyun
# because the user is typically in China (this repo is built around
# hub.rat.dev which is a Chinese registry mirror). Override at build time:
#   docker build --build-arg APT_MIRROR= -t ...        (use upstream Ubuntu)
#   docker build --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn ...
ARG APT_MIRROR=https://mirrors.aliyun.com
RUN set -eux; \
    if [ -n "${APT_MIRROR}" ]; then \
        if [ ! -f /etc/apt/sources.list.bak ]; then \
            cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
        fi; \
        sed -i \
          -e "s#http://archive.ubuntu.com/ubuntu/#${APT_MIRROR}/ubuntu/#g" \
          -e "s#http://security.ubuntu.com/ubuntu/#${APT_MIRROR}/ubuntu/#g" \
          -e "s#http://ports.ubuntu.com/ubuntu-ports/#${APT_MIRROR}/ubuntu-ports/#g" \
          /etc/apt/sources.list; \
    fi; \
    apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update; \
    apt-get install -y --no-install-recommends \
      bash \
      bzip2 \
      ca-certificates \
      curl \
      fontconfig \
      git \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      tini \
      wget \
      # --- basic utilities ---
      bc \
      debianutils \
      file \
      htop \
      jq \
      less \
      lsof \
      nano \
      procps \
      psmisc \
      rsync \
      tree \
      unzip \
      vim-tiny \
      zip \
      # --- network diagnostics ---
      dnsutils \
      iproute2 \
      iputils-ping \
      iputils-tracepath \
      net-tools \
      netcat-openbsd \
      telnet \
      traceroute \
    ; \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL ${MINIFORGE_URL} -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda config --system --set channel_priority strict \
    && conda config --system --set channel_alias https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \
    && conda clean -afy

WORKDIR /opt/PocketXMol

RUN printf '%s\n' \
      'name: pxm_cu128' \
      'channels:' \
      '  - conda-forge' \
      '  - bioconda' \
      'dependencies:' \
      '  - python=3.10' \
      '  - pip=24.3.1' \
      '  - easydict=1.9' \
      '  - numpy=1.24' \
      '  - openbabel=3.1.1' \
      '  - pandas=1.5.2' \
      '  - pyyaml=6.0.2' \
      '  - tqdm=4.64.0' \
      '  - rdkit=2023.9.3' \
      '  - biopython=1.83' \
      '  - networkx=2.8' \
      '  - setuptools<70' \
      '  - scikit-learn=1.1.0' \
      '  - scipy=1.10.1' \
      '  - seaborn=0.12.1' \
      '  - tensorboard=2.20.0' \
      '  - pip:' \
      '      - lmdb==1.2.1' \
      '      - peptidebuilder==1.1.0' \
    > /tmp/environment_pxm_cu128.yml \
    && conda env create -f /tmp/environment_pxm_cu128.yml \
    && conda clean -afy

# Use Tsinghua as the primary index (China CDN, hosts cu128 torch wheels too).
# pytorch.org is kept as a fallback for any package Tsinghua doesn't have.
# data.pyg.org is needed for torch-scatter/sparse/cluster/spline_conv/pyg_lib
# wheels built against torch-2.8.0+cu128, which neither PyPI nor Tsinghua
# mirror cleanly (the build tag in the wheel filename changes per torch+CUDA
# combo, so we pin to the PyG find-links page).
RUN python -m pip install --no-cache-dir \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
      lightning==2.6.5 \
      pytorch-lightning==2.6.5 \
      torch==2.8.0 \
      torch-geometric==2.8.0 \
      torch-cluster==1.6.3+pt28cu128 \
      torch-scatter==2.1.2+pt28cu128 \
      torch-sparse==0.6.18+pt28cu128 \
      torch-spline-conv==1.2.2+pt28cu128 \
      pyg-lib==0.6.0+pt28cu128

RUN mkdir -p /tmp/matplotlib /tmp/.cache/fontconfig \
    && fc-cache -f

# Pre-clone PocketXNA (aptamer / nucleic-acid design, fork of PocketXMol)
# into the image so the training scripts and configs are immediately available
# at /opt/PocketXNA without needing creds inside the container at runtime.
ARG POCKETXNA_REPO=https://github.com/mmjwxbc/PocketXNA.git
ARG POCKETXNA_DIR=/opt/PocketXNA
RUN git clone --depth 1 "${POCKETXNA_REPO}" "${POCKETXNA_DIR}"

RUN set -eux; \
    if [ -n "${APT_MIRROR}" ] && [ ! -f /etc/apt/sources.list.bak ]; then \
        cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
    fi; \
    if [ -n "${APT_MIRROR}" ] && [ -f /etc/apt/sources.list.bak ]; then \
        sed -i \
          -e "s#http://archive.ubuntu.com/ubuntu/#${APT_MIRROR}/ubuntu/#g" \
          -e "s#http://security.ubuntu.com/ubuntu/#${APT_MIRROR}/ubuntu/#g" \
          /etc/apt/sources.list; \
    fi; \
    apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update; \
    apt-get install -y --no-install-recommends \
      openssh-server \
      openssh-client \
    ; \
    mkdir -p /run/sshd; \
    ssh-keygen -A; \
    sed -i \
      -e 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' \
      -e 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' \
      -e 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' \
      /etc/ssh/sshd_config; \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
