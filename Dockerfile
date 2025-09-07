# Use NVIDIA's official CUDA base image for Ubuntu 22.04 and CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GROMACS_VERSION=2024.3
ENV PLUMED_VERSION=2.9.3
ENV PATH="/usr/local/gromacs/bin:/usr/local/plumed/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/gromacs/lib:/usr/local/plumed/lib:${LD_LIBRARY_PATH:-}"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    fftw3-dev \
    libfftw3-dev \
    libopenmpi-dev \
    openmpi-bin \
    mpi-default-bin \
    mpi-default-dev \
    perl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build and install PLUMED
WORKDIR /tmp
RUN wget https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-${PLUMED_VERSION}.tgz && \
    tar -xzf plumed-${PLUMED_VERSION}.tgz && \
    rm plumed-${PLUMED_VERSION}.tgz

WORKDIR /tmp/plumed-${PLUMED_VERSION}
RUN ./configure --enable-modules=all --prefix=/usr/local/plumed && make -j$(nproc) && make install

RUN /usr/local/plumed/bin/plumed --help

# Build and install GROMACS with GPU and MPI
WORKDIR /tmp
RUN wget http://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz && \
    tar -xzf gromacs-${GROMACS_VERSION}.tar.gz && \
    rm gromacs-${GROMACS_VERSION}.tar.gz

WORKDIR /tmp/gromacs-${GROMACS_VERSION}
RUN plumed patch -p -e gromacs-${GROMACS_VERSION}

WORKDIR /tmp/gromacs-${GROMACS_VERSION}/build
RUN cmake .. \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DGMX_MPI=ON \
    -DGMX_GPU=CUDA \
    -DPLUMED_ROOT=/usr/local/plumed \
    -DGMX_PLUMED=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs && \
    make -j$(nproc) && make install

WORKDIR /
RUN rm -rf /tmp/gromacs-${GROMACS_VERSION} /tmp/plumed-${PLUMED_VERSION}

# Create a non-root user with sudo privileges
ARG USERNAME=spal
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && apt-get install -y sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the non-root user
USER $USERNAME
WORKDIR /home/$USERNAME



CMD ["bash", "-c", "source /usr/local/gromacs/bin/GMXRC && gmx --version"]
