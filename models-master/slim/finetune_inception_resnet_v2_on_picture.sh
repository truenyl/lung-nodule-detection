#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_flowers.sh

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/truenyl/try_tensorflow/task/inception_resnet_v2_checkpoint

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/truenyl/try_tensorflow/task/inception_resnet_v2_model

# Where the dataset is saved to.
DATASET_DIR=/home/truenyl/try_tensorflow/task/Alltfrecord

# Download the pre-trained checkpoint.
#if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
#  mkdir ${PRETRAINED_CHECKPOINT_DIR}
#fi
#if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
#  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
#  tar -xvf inception_v3_2016_08_28.tar.gz
#  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
#  rm inception_v3_2016_08_28.tar.gz
#fi

# Download the dataset

#python download_and_convert_data.py \
#  --dataset_name=flowers \
#  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=picture \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_resnet_v2 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

#print("Fine-tune the new layers for 1000 steps successfully!")

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=picture \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_resnet_v2

#print("Run evalution successfully!")

# Fine-tune all the new layers for 500 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=picture \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_resnet_v2 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

#print("Fine-tune all the new layers for 500 steps successfully!")

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=picture \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_resnet_v2

#print("Run evalution successfully!")
