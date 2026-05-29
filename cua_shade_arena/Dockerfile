FROM aisiuk/inspect-tool-support

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    bash-completion \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    tar \
    zip \
    unzip \
    curl \
    wget \
    less \
    git \
    vim \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    ipython \
    pandas \
    numpy \
    requests \
    openpyxl

RUN mkdir -p /inputs && \
    chmod -R 777 /inputs

COPY tac-openhands /inputs