#!/usr/bin/env bash
ROOT_DIR=$(pwd)
declare -a DIRS=( "pointmvsnet/functions" )
echo "ROOT_DIR=${ROOT_DIR}"

for BUILD_DIR in "${DIRS[@]}"
do
    echo "BUILD_DIR=${BUILD_DIR}"
    cd $BUILD_DIR
    if [ -d "build" ]; then
        rm -r build
    fi
    python setup.py build_ext --inplace
    cd $ROOT_DIR
done