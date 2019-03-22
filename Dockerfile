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
RUN pip3 install numpy pandas torch scipy livelossplot Cython requests

# Install lie_learn
# see also https://github.com/jonas-koehler/s2cnn
RUN git clone https://github.com/AMLab-Amsterdam/lie_learn /opt/lie_learn
RUN pip3 install /opt/lie_learn

# Install SE3CNN
RUN git clone -b missing_point https://github.com/blondegeek/se3cnn.git /opt/se3cnn
RUN pip3 install /opt/se3cnn

COPY . /workspace/
WORKDIR /workspace/

# Launch Notebook server
EXPOSE 8888
CMD ["jupyter-notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]

#EOF