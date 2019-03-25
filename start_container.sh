#!/bin/bash -e

set -x

sudo docker build -t pao .

sudo docker run --init -ti -p 8888:8888 -v $PWD:/workspace pao

#EOF