#!/bin/bash -e

set -x

$1 docker build -t pao .
$1 docker run --init -ti -p 127.0.0.1:8888:8888 -v $PWD:/workspace pao

#EOF
