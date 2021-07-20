#!/bin/bash
set -e

BASE_MODEL_NAME=$1
WEIGHTS_FILE1=$2
WEIGHTS_FILE2=$3
IMAGE_SOURCE=$4

# predict
python -m evaluater.predict_both \
--base-model-name $BASE_MODEL_NAME \
--weights-file1 $WEIGHTS_FILE1 \
--weights-file2 $WEIGHTS_FILE2 \
--image-source $IMAGE_SOURCE
