.#!/usr/bin/env bash
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

# BZ PSG GD BZD
# 64 100  8  12
# 64  50  4  12
# 64  20  2  16
# 64  10  1  16

# sh ./scripts/answer_prediction/train_aq.sh out_data/answer-prediction-nq/best-model.pt 

CKPT_ALL=$1

BZ=64
GD=16
PSG=8
BZP=8
PSG_DIR=$2
NUM_EPOCH=30
EP=30

python cli.py \
--task=qa \
--train_file=ambigqa/train.json \
--predict_file=ambigqa/dev.json \
--output_dir=answer_prediction_aq_rfid \
--do_train=True \
--ambigqa=True \
--wiki_2020=True \
--bert_name=bart-large \
--checkpoint=${CKPT_ALL} \
--max_cat_answer_length=48 \
--max_answer_length=64 \
--n_jobs=96 \
--psg_sel_dir=${PSG_DIR} \
--top_k_passages=${PSG} \
--use_reranker=True \
--decoder_start_token_id=2 \
--train_batch_size=${BZ} \
--predict_batch_size=${BZP} \
--learning_rate=3e-5 \
--warmup_proportion=0.1 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--gradient_accumulation_steps=${GD} \
--num_train_epochs=${NUM_EPOCH} \
--wait_step=1000000 \
--verbose=True \
--eval_period=${EP} \
--use_gpu_ids=0,1,2,3 \
--discard_not_found_answers=True