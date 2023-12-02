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


import string
import numpy as np
import time
import regex
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.utils.data.distributed

class MySimpleQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False,
                 answer_as_prefix=False,
                 tokenizer=None,
                 answers=None,
                ):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if is_training and out_metadata is None else out_metadata
        self.is_training = is_training
        self.answer_as_prefix = answer_as_prefix
        self.answers = answers
        self.tokenizer = tokenizer

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            if self.answer_as_prefix:
                out_idx = self.out_metadata[idx][0]
                return self.input_ids[idx], self.attention_mask[idx], \
                    self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx])) # XXX: okay to just choose one?
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        golden = self._get_golden_answer(self.input_ids[in_idx], self.decoder_input_ids[out_idx])
        golden = torch.tensor(golden, dtype=torch.int64)
    
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], golden
    
    def _get_golden_answer(self, multiple_qp_ids, answer_ids):
        golden = []
        
        # get str answers
        bos_idx = (answer_ids == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0][0]
        eos_idx = (answer_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
        str_answers = self.tokenizer.decode(answer_ids[bos_idx+1:eos_idx])
        
        for qp_ids in multiple_qp_ids:
            # get str passages
            eos_idx = (qp_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
            curr_passage_id = qp_ids[eos_idx+1:]
            str_passages = self.tokenizer.decode(curr_passage_id)
            
            is_golden = self._check_answers(str_answers, str_passages)

            golden.append(is_golden)
        
        return golden

    def _check_answers(self, answers, passage):
        """Copied from  RFiD"""
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        passage = white_space_fix(remove_articles(remove_punc(lower(passage.strip()))))

        preprocessed_answer = white_space_fix(remove_articles(remove_punc(lower(answers.strip()))))
        if preprocessed_answer in passage:
            return 1
        return 0


class MySimpleQALMFilteringDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False,
                 answer_as_prefix=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if is_training and out_metadata is None else out_metadata
        self.is_training = is_training
        self.answer_as_prefix = answer_as_prefix

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        assert not self.is_training
        idx = self.in_metadata[idx][0]
        return self.input_ids[idx], self.attention_mask[idx], \
               self.decoder_input_ids[idx], self.decoder_attention_mask[idx]


class MySimpleQADatasetForPair(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None, metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.metadata = metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)
        assert not self.is_training or len(self.input_ids)==len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.metadata[-1][-1]
        assert self.metadata[-1][-1]==len(self.input_ids)

    def __len__(self):
        return len(self.metadata) if self.is_training else len(self.input_ids)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]
        idx = np.random.choice(range(*self.metadata[idx]))
        return self.input_ids[idx], self.attention_mask[idx], \
            self.decoder_input_ids[idx], self.decoder_attention_mask[idx]


class MySimpleQGDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) if is_training and out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]==len(self.in_metadata)==len(self.out_metadata)

    def __len__(self):
        if self.is_training:
            return len(self.out_metadata)
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MySimpleQGWeightedLossDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False,
                 weighted_position=None):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) if is_training and out_metadata is None else out_metadata
        self.is_training = is_training
        self.weighted_position = None if weighted_position is None else torch.LongTensor(weighted_position)

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]==len(self.in_metadata)==len(self.out_metadata)

    def __len__(self):
        if self.is_training:
            return len(self.out_metadata)
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], self.weighted_position[out_idx]


class MySimpleQGDynamicDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False, discard_not_found_answers=None):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) if is_training and out_metadata is None else out_metadata
        self.is_training = is_training
        self.discard_not_found_answers = discard_not_found_answers

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]==len(self.in_metadata)==len(self.out_metadata)

    def __len__(self):
        if self.is_training:
            return len(self.out_metadata)
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], sum(self.discard_not_found_answers[idx])

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], sum(self.discard_not_found_answers[in_idx])


class MySimpleQGDynamicWeightedLossDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False, discard_not_found_answers=None,
                 weighted_position=None):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) if is_training and out_metadata is None else out_metadata
        self.is_training = is_training
        self.discard_not_found_answers = discard_not_found_answers
        self.weighted_position = None if weighted_position is None else torch.LongTensor(weighted_position)

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]==len(self.in_metadata)==len(self.out_metadata)

    def __len__(self):
        if self.is_training:
            return len(self.out_metadata)
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], sum(self.discard_not_found_answers[idx])

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], \
               sum(self.discard_not_found_answers[in_idx]), self.weighted_position[out_idx]


class MyQADataset(Dataset):
    def __init__(self, data,
                 is_training=False, train_M=None, test_M=None):
        self.data = data #.dictionify()
        self.positive_input_ids = self.tensorize("positive_input_ids")
        self.positive_input_mask = self.tensorize("positive_input_mask")
        self.positive_token_type_ids = self.tensorize("positive_token_type_ids")
        assert len(self.positive_input_ids)==len(self.positive_input_mask)==len(self.positive_token_type_ids)

        if is_training:
            self.positive_start_positions = self.tensorize("positive_start_positions")
            self.positive_end_positions = self.tensorize("positive_end_positions")
            self.positive_answer_mask = self.tensorize("positive_answer_mask")
            self.negative_input_ids = self.tensorize("negative_input_ids")
            self.negative_input_mask = self.tensorize("negative_input_mask")
            self.negative_token_type_ids = self.tensorize("negative_token_type_ids")
            assert len(self.negative_input_ids)==len(self.negative_input_mask)==len(self.negative_token_type_ids)
            assert len(self.positive_input_ids)==\
                    len(self.positive_start_positions)==len(self.positive_end_positions)==len(self.positive_answer_mask)
            assert all([len(positive_input_ids)>0 for positive_input_ids in self.positive_input_ids])

        self.is_training = is_training
        self.train_M = train_M
        self.test_M = test_M

    def __len__(self):
        return len(self.positive_input_ids)

    def __getitem__(self, idx):
        if not self.is_training:
            input_ids = self.positive_input_ids[idx][:self.test_M]
            input_mask = self.positive_input_mask[idx][:self.test_M]
            token_type_ids = self.positive_token_type_ids[idx][:self.test_M]
            return [self._pad(t, self.test_M) for t in [input_ids, input_mask, token_type_ids]]

        # sample positive
        positive_idx = np.random.choice(len(self.positive_input_ids[idx]))
        #positive_idx = 0
        positive_input_ids = self.positive_input_ids[idx][positive_idx]
        positive_input_mask = self.positive_input_mask[idx][positive_idx]
        positive_token_type_ids = self.positive_token_type_ids[idx][positive_idx]
        positive_start_positions = self.positive_start_positions[idx][positive_idx]
        positive_end_positions = self.positive_end_positions[idx][positive_idx]
        positive_answer_mask = self.positive_answer_mask[idx][positive_idx]

        # sample negatives
        negative_idxs = np.random.permutation(range(len(self.negative_input_ids[idx])))[:self.train_M-1]
        negative_input_ids = [self.negative_input_ids[idx][i] for i in negative_idxs]
        negative_input_mask = [self.negative_input_mask[idx][i] for i in negative_idxs]
        negative_token_type_ids = [self.negative_token_type_ids[idx][i] for i in negative_idxs]
        negative_input_ids, negative_input_mask, negative_token_type_ids = \
            [self._pad(t, self.train_M-1) for t in [negative_input_ids, negative_input_mask, negative_token_type_ids]]

        # aggregate
        input_ids = torch.cat([positive_input_ids.unsqueeze(0), negative_input_ids], dim=0)
        input_mask = torch.cat([positive_input_mask.unsqueeze(0), negative_input_mask], dim=0)
        token_type_ids = torch.cat([positive_token_type_ids.unsqueeze(0), negative_token_type_ids], dim=0)
        start_positions, end_positions, answer_mask = \
            [self._pad([t], self.train_M) for t in [positive_start_positions,
                                                  positive_end_positions,
                                                  positive_answer_mask]]

        # golden
        
        
        return input_ids, input_mask, token_type_ids, start_positions, end_positions, answer_mask

    def tensorize(self, key):
        return [torch.LongTensor(t) for t in self.data[key]] if key in self.data.keys() else None

    def _pad(self, input_ids, M):
        if len(input_ids)==0:
            return torch.zeros((M, self.negative_input_ids[0].size(1)), dtype=torch.long)
        if type(input_ids)==list:
            input_ids = torch.stack(input_ids)
        if len(input_ids)==M:
            return input_ids
        return torch.cat([input_ids,
                          torch.zeros((M-input_ids.size(0), input_ids.size(1)), dtype=torch.long)],
                         dim=0)


class MyQAGenDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1 + len(input_ids)))) if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1 + len(decoder_input_ids)))) if is_training and out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids) == len(self.attention_mask) == self.in_metadata[-1][-1]
        assert not self.is_training or len(self.decoder_input_ids) == len(self.decoder_attention_mask) == self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
               self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MyRerankerDataset(Dataset):
    def __init__(self, data,
                 is_training=False, train_MP=None, train_MN=None, test_M=None):
        self.data = data #.dictionify()
        self.positive_input_ids = self.tensorize("positive_input_ids")
        self.positive_input_mask = self.tensorize("positive_input_mask")
        self.positive_token_type_ids = self.tensorize("positive_token_type_ids")
        assert len(self.positive_input_ids)==len(self.positive_input_mask)==len(self.positive_token_type_ids)

        if is_training:
            self.positive_start_positions = self.tensorize("positive_start_positions")
            self.positive_end_positions = self.tensorize("positive_end_positions")
            self.positive_answer_mask = self.tensorize("positive_answer_mask")
            self.negative_input_ids = self.tensorize("negative_input_ids")
            self.negative_input_mask = self.tensorize("negative_input_mask")
            self.negative_token_type_ids = self.tensorize("negative_token_type_ids")
            assert len(self.negative_input_ids)==len(self.negative_input_mask)==len(self.negative_token_type_ids)
            assert all([len(positive_input_ids)>0 for positive_input_ids in self.positive_input_ids])

        self.is_training = is_training
        self.train_MP = train_MP
        self.train_MN = train_MN
        self.test_M = test_M

    def __len__(self):
        return len(self.positive_input_ids)

    def __getitem__(self, idx):
        if not self.is_training:
            input_ids = self.positive_input_ids[idx][:self.test_M]
            input_mask = self.positive_input_mask[idx][:self.test_M]
            token_type_ids = self.positive_token_type_ids[idx][:self.test_M]
            return [self._pad(t, self.test_M) for t in [input_ids, input_mask, token_type_ids]]

        # sample positive
        positive_idxs = np.random.permutation(range(len(self.positive_input_ids[idx])))[:self.train_MP]
        positive_input_ids = [self.positive_input_ids[idx][i] for i in positive_idxs]
        positive_input_mask = [self.positive_input_mask[idx][i] for i in positive_idxs]
        positive_token_type_ids = [self.positive_token_type_ids[idx][i] for i in positive_idxs]
        labels = torch.LongTensor(list(range(len(positive_idxs))) + [-1] * (self.train_MP - len(positive_idxs)))
        positive_input_ids, positive_input_mask, positive_token_type_ids = \
            [self._pad(t, self.train_MP) for t in [positive_input_ids, positive_input_mask, positive_token_type_ids]]

        # sample negatives
        negative_idxs = np.random.permutation(range(len(self.negative_input_ids[idx])))[:self.train_MN]
        negative_input_ids = [self.negative_input_ids[idx][i] for i in negative_idxs]
        negative_input_mask = [self.negative_input_mask[idx][i] for i in negative_idxs]
        negative_token_type_ids = [self.negative_token_type_ids[idx][i] for i in negative_idxs]
        negative_input_ids, negative_input_mask, negative_token_type_ids = \
            [self._pad(t, self.train_MN) for t in [negative_input_ids, negative_input_mask, negative_token_type_ids]]

        # aggregate
        input_ids = torch.cat([positive_input_ids, negative_input_ids], dim=0)
        input_mask = torch.cat([positive_input_mask, negative_input_mask], dim=0)
        token_type_ids = torch.cat([positive_token_type_ids, negative_token_type_ids], dim=0)

        return input_ids, input_mask, token_type_ids, labels

    def tensorize(self, key):
        return [torch.LongTensor(t) for t in self.data[key]] if key in self.data.keys() else None

    def _pad(self, input_ids, M):
        if len(input_ids)==0:
            return torch.zeros((M, self.negative_input_ids[0].size(1)), dtype=torch.long)
        if type(input_ids)==list:
            input_ids = torch.stack(input_ids)
        if len(input_ids)==M:
            return input_ids
        return torch.cat([input_ids,
                          torch.zeros((M-input_ids.size(0), input_ids.size(1)), dtype=torch.long)],
                         dim=0)


class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training, batch_size=None, **kwargs):
        if is_training:
            sampler = RandomSampler(dataset) if args.is_distributed == 0 else torch.utils.data.distributed.DistributedSampler(dataset)
            batch_size = args.train_batch_size if batch_size is None else batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size if batch_size is None else batch_size

        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class MyQAGenDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training, batch_size=None, **kwargs):
        if is_training:
            sampler = RandomSampler(dataset) if args.is_distributed == 0 else torch.utils.data.distributed.DistributedSampler(dataset)
            batch_size = batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size if batch_size is None else batch_size

        super(MyQAGenDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, **kwargs)