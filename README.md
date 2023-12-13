# Mitigating Generator Input Spuriousness For Multi-Answer Open Domain Question Answering

This repository is the implementation, based on [REFUEL paper's official repository](https://github.com/given131/refuel), of research project for 2023 Fall Semester NLP class. We are Team 9 (Jaeyoung Lee, Junggeun Do). 

## Requirements
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

We use 4 RTX2080Ti (11GB) GPUs to run the following experiments.

## Content
- [Overview](#overview)
- [Resources](#resources)
- [Retrieval \& Reranking](#retrieval-and-reranking)
- [Multiple Answer Prediction](#answer-prediction)


### TL;DR
After preparing the [Resources](#resources) and performing [Retrieval \& Reranking](#retrieval-and-reranking), we trained our baseline and proposed models using the code similar to the one provided below. 

This code serves as a pseudocode example, demonstrating implementation details through key configurations. 

```
# Pre-train multiple answer prediction models on Natrual Questions
python cli.py \ 
--task=qa \
--bert_name=bart-large \
--top_k_passages=8 \
--num_train_epochs=5
--train_batch_size=64 \
--gradient_accumulation_steps=16 \
--predict_batch_size=8 \
--learning_rate=5e-5 \
--warmup_proportion=0.1 \
--weight_decay=0.01 \
--max_grad_norm=1.0 
# add --use_classifier option to train proposed model

# Fine-tune multiple answer prediction models AmbigNQ
python cli.py \
--task=qa \
--bert_name=bart-large \
--top_k_passages=8 \
--num_train_epochs=15 \
--train_batch_size=64 \
--gradient_accumulation_steps=16 \
--predict_batch_size=8 \
--learning_rate=3e-5 \
--warmup_proportion=0.1 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--max_cat_answer_length=48 \
--max_answer_length=64 \
--ambigqa=True \
--wiki_2020=True
# add --use_classifier option to train proposed model
```


### Overview
Our solution is a pipelined approach which contains the following steps:

1) Given an ambiguous prompt question, the retrieval module firstly retrieves 1000 question-relevant passages;

2) The reranking module reranks 1000 passages by interacting the prompt question with each passage;

3) The answer prediction module takes top 100 reranked passages, and generate one or multiple answers.

### Resources
Download the following datasets and checkpoints

-  Wikipedia corpus for NQ-open and AmbigQA (The questions in these two datasets are annotated by difference Wikipedia dumps):

```
# Wikipedia corpus for NQ-open
python download_data.py \
    --resource data.wikipedia_split.psgs_w100 \
    --output_dir retriever_data/wikipedia_split/

# Wikipedia corpus for AmbigQA
python download_data.py \
    --resource data.wikipedia_split.psgs_w100_20200201 \
    --output_dir retriever_data/wikipedia_split/
```

- NQ-open dataset, AmbigQA dataset, and official Natural Question answers ([why?](https://github.com/shmsw25/AmbigQA/blob/master/codes/README.md): see "Note" there)

```
# AmbigQA dataset
wget https://nlp.cs.washington.edu/ambigqa/data/ambignq.zip -O reader_data/ambigqa/

# NQ-open dataset
python download_data.py \
    --resource data.nqopen.{train,dev,test} \
    --output_dir reader_data/nqopen/
python download_data.py \
    --resource data.nqopen.{train,dev,test}_id2answers \
    --output_dir reader_data/nqopen/
```

### Retrieval and Reranking

#### Retrieval 

We use the Dense Passage Retriever [1] as our retriever. There two versions of DPR checkpoints, one is trained only on the NQ-open dataset (single.pt), another is jointly trained on five QA datasets (multiset.pt). According to our experiments, `multiset.pt` performs slightly better than the `single.pt`. So here we only use the multiset version.

- Download DPR checkpoints

```
python download_data.py \
    --resource checkpoint.retriever.multiset.bert-base-encoder \
    --output_dir retriever_data/checkpoint/retriever/multiset/
```

- Encode all passages in Wikipedia into d-dimensional dense representations (it may take several hours to finish this step)

The wikipedia is splitted into 10 shards for encoding. Here we use a for loop to encode the dense representations, it would be better to split these cmds into different GPUs to save time.

```
# For NQ-open Wikipedia Dump

for i in 0 1 2 3 4 5 6 7 8 9
do
    ./scripts/retriever/generate_dense_representation_nq.sh $i $GPU_ID
done

# For AmbigQA Wikipedia Dump
for i in 0 1 2 3 4 5 6 7 8 9
do
    ./scripts/retriever/generate_dense_representation_aq.sh $i $GPU_ID
done
```

- Retrieve 1000 relevant passages for questions in NQ-open / AmbigQA: (still, it may take several hours to finish this step)
```
# NQ-open
./scripts/retriever/retrieve_psgs_nq.sh {train,dev,test} $GPU_ID

# AmbigQA
./scripts/retriever/retrieve_psgs_aq.sh {train,dev} $GPU_ID

# Leaderboard Submission
./scripts/retriever/retrieve_psgs_aq_leaderboard.sh $GPU_ID
```

#### Reranking
We train a `bert-large-uncased`-based reranker with listwise ranking loss. It takes ~1 day to finish training. 

Firstly, we train the reranker on the NQ-open dataset:

```
./scripts/reranker/train_nq.sh
```

Then, use the trained reranker to rerank passages for train, dev, test set of NQ-Open.

```
./script/reranker/inference_nq.sh path/to/saved/model.pt {train|dev|test}

```

### Multiple Answer Prediction
We firstly pre-train the multiple answer prediction model on NQ-open, and fine-tune it on AmbigQA.

It takes 1~2 days for pre-training and about 1 day for fine-tuning.

- Pre-train on NQ-open

```
./scripts/answer_prediction/train_nq.sh path/to/reranker/outputs
```

- Evaluate for NQ-open

```
./scripts/answer_prediction/inference_nq.sh $GPU_ID {dev|test} path/to/model path/to/reranker/outputs
```

- Fine-tune on AmbigQA

```
./scripts/answer_prediction/train_aq.sh path/to/pretrained/ckpt path/to/reranker/outputs
```
- Evaluate for Fine-tuned model

```
./scripts/answer_prediction/inference_aq.sh $GPU_ID path/to/model/ckpt path/to/reranker/outputs
```

## References
1. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Karpukhin et al., EMNLP 2020.
2. [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282), Gautier Izacard, Edouard Grave, ArXiv, July 2020.
3. [AmbigQA: Answering Ambiguous Open-domain Questions](https://arxiv.org/abs/2004.10645), Min et al., EMNLP 2020.
4. [Answering Ambiguous Questions through Generative Evidence Fusion and Round-Trip Prediction](https://aclanthology.org/2021.acl-long.253/), Gao et al., ACL-IJCNLP 2021.
5. [RFiD: Towards Rational Fusion-in-Decoder for Open-Domain Question Answering](https://aclanthology.org/2023.findings-acl.155/), Wang et al., ACL Findings 2023.
