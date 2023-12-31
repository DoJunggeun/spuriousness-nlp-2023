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


import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel

class MyBiEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.ctx_model = BertModel(config)
        self.question_model = BertModel(config)
        self.init_weights()

    def forward_qbert(self, input_ids, attention_mask):
        return self.question_model(input_ids=input_ids, attention_mask=attention_mask)[0][:,0,:]

    def forward_pbert(self, input_ids, attention_mask):
        return self.ctx_model(input_ids=input_ids, attention_mask=attention_mask)[0][:,0,:]

    def forward(self,
                q_input_ids, q_attention_mask,
                p_input_ids, p_attention_mask,
                is_training=False):
        '''
        :q_input_ids, q_attention_mask, q_token_type_ids: [N, L]
        :p_input_ids, p_attention_mask, p_token_type_ids: [N, M, L]
        '''
        N, M, L = p_input_ids.size()
        question_output = self.forward_qbert(q_input_ids, q_attention_mask)
        passage_output = self.forward_pbert(input_ids=p_input_ids.view(-1, L),
                                            attention_mask=p_attention_mask.view(-1, L))
        if is_training:
            inner_prods = torch.matmul(question_output, passage_output.transpose(0, 1))
            loss_fct = CrossEntropyLoss()
            labels = M * torch.arange(N, dtype=torch.long).cuda()
            total_loss = loss_fct(inner_prods, labels) # [N, N*M]
            return total_loss
        else:
            return question_output, passage_output


