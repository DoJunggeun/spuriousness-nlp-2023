#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


BZ=64 # 64
TK=8 # 100
GD=16 # 8
LR=5e-5 # for bz 64
EP=200 # 600
BZD=8 # 12
PSG_DIR=$1
NUM_EPOCH=10 # 10

# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 cli.py \
# OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 cli.py \
# NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=4 torchrun --standalone --nnodes=1 --nproc_per_node=4 cli.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 cli.py \
# --is_distributed 1 --num_gpus 4 --average_gradient 1 \


# replace python cli.py to these two lines 
# OMP_NUM_THREADS=4 
python cli.py \
--task=qa \
--train_file=nqopen/train.json \
--predict_file=nqopen/dev.json \
--do_train=True \
--checkpoint=out_data/answer-prediction-nq-baseline/model-step2800.pt \
--start_step 2800 \
--output_dir=answer-prediction-nq-baseline \
--bert_name=bart-large \
--max_answer_length=16 \
--psg_sel_dir=${PSG_DIR} \
--discard_not_found_answer=True \
--top_k_passages=${TK} \
--use_reranker=True \
--decoder_start_token_id=2 \
--train_batch_size=${BZ} \
--predict_batch_size=${BZD} \
--learning_rate=${LR} \
--warmup_proportion=0.1 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--gradient_accumulation_steps=${GD} \
--num_train_epochs=${NUM_EPOCH} \
--wait_step=100000000 \
--verbose=True \
--eval_period=${EP} \
--n_jobs=96 \
--use_gpu_ids=0,1,2,3


