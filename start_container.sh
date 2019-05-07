#!/bin/bash -e

set -x

docker build -t pao .

docker run --init -ti -p 8888:8888 -v $PWD:/workspace pao

#EOF
