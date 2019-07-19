FROM ubuntu:18.04
USER root

#
#
#   This Dockerfile is mainly meant for developing and testing:
#     1. docker build --tag pao ./
#     2. docker run --init -ti -p8888:8888 pao
#     3. open http://localhost:8888/
#
#

# Install Ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential       \
    python3-dev           \
    python3-setuptools    \
    python3-wheel         \
    python3-pip           \
    less                  \
    nano                  \
    sudo                  \
    git                   \
    npm                   \
  && rm -rf /var/lib/apt/lists/*

# Install Jupyter
RUN pip3 install notebook

# Install more Python packages
RUN pip3 install numpy pandas torch scipy livelossplot Cython requests ase

# Install Tensorflow
#RUN pip3 install tensorflow

# Tensorflow 2
#https://www.tensorflow.org/alpha/guide/effective_tf2
#https://www.tensorflow.org/alpha/guide/migration_guide
RUN pip3 install tensorflow==2.0.0-alpha0
RUN pip3 install sympy

# Install lie_learn
# see also https://github.com/jonas-koehler/s2cnn
RUN git clone https://github.com/AMLab-Amsterdam/lie_learn /opt/lie_learn

#Fix for Numpy v1.16.3 changes
#COPY ./fix_lie_learn_allow_pickle.patch /tmp/
#RUN cd /opt/lie_learn; cat /tmp/fix_lie_learn_allow_pickle.patch | patch -p1

RUN pip3 install /opt/lie_learn

## Install SE3CNN
#RUN git clone -b missing_point https://github.com/blondegeek/se3cnn.git /opt/se3cnn
#RUN pip3 install /opt/se3cnn
#
## Install CP2K
#RUN git clone -b pao_ab https://github.com/oschuett/cp2k.git /opt/cp2k
#
#RUN apt-get update -qq && apt-get install -qq --no-install-recommends  \
#    autoconf                               \
#    autogen                                \
#    automake                               \
#    autotools-dev                          \
#    ca-certificates                        \
#    cmake                                  \
#    git                                    \
#    less                                   \
#    libtool                                \
#    make                                   \
#    nano                                   \
#    pkg-config                             \
#    python                                 \
#    rsync                                  \
#    unzip                                  \
#    wget                                   \
#    gcc                     \
#    g++                     \
#    gfortran                \
#    fftw3-dev               \
#    libopenblas-dev         \
#    liblapack-dev           \
#  && rm -rf /var/lib/apt/lists/*
#
#WORKDIR /opt/cp2k/tools/toolchain/
#RUN ./install_cp2k_toolchain.sh   \
#        --mpi-mode=no             \
#        --with-gcc=system         \
#        --with-cmake=system       \
#        --with-fftw=system        \
#        --with-openblas=system    \
#        --with-reflapack=system   \
#        --with-libxsmm=install    \
#        --with-libint=no          \
#        --with-libxc=no           \
#        --with-sirius=no          \
#        --with-gsl=no             \
#        --with-hdf5=no            \
#        --with-spglib=no          \
#        --with-libvdwxc=no        \
#        --with-json-fortran=no    \
#  && rm -rf ./build
#
#WORKDIR /opt/cp2k
#RUN git submodule update --init --recursive
#
#WORKDIR /opt/cp2k/arch
#RUN ln -vs ../tools/toolchain/install/arch/* .
#
#WORKDIR /opt/cp2k/
#RUN bash -c "source ./tools/toolchain/install/setup && make -j VERSION=ssmp"

#COPY . /workspace/ # using volume mount instead

WORKDIR /workspace/

# Launch Notebook server
EXPOSE 8888
CMD ["jupyter-notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]

#EOF