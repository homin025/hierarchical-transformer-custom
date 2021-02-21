# Hierarchical Transformer Custom


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Reference](#reference)

## Background

This repository is referenced from the paper ['Hierarchical Transformers for Multi-Document Summarization'](https://arxiv.org/pdf/1905.13164.pdf).

It is almost the same with referenced code, but I rewrote this for reviewing and understanding code. It also uses korean vocabulary to apply Korean news for multi-document summarization

## Install

This project uses [pytorch](https://pytorch.org/).

```sh
$ pip install -r requirements.txt
```

## Usage

For Training

```sh
python main.py -mode train -data_path ./data/train.json -batch_size 100 -seed 666 -train_steps 1000000 -save_checkpoint_steps 100000 -report_every 1000 -trunc_tgt_ntoken 600 -trunc_src_nblock 24 -accum_count 4 -dec_dropout 0.1 -enc_dropout 0.1 -label_smoothing 0.1 -vocab_path ./tokenizer/korean_32000.model -model_path ./model/ -accum_count 4 -log_file ./log.txt -inter_layer 6,7 -inter_heads 8 -hier -world_size 3 -visible_gpus 0,1,2 -gpu_rank 0,1,2 -train_from ./model/model.pt
```

For Testing

```sh
python main.py -mode test -data_path ./data/test.json -batch_size 1 -seed 6666 -train_steps 1 -save_checkpoint_steps 1 -report_every 1 -trunc_tgt_ntoken 600 -trunc_src_nblock 24 -accum_count 4 -dec_dropout 0.1 -enc_dropout 0.1 -label_smoothing 0.1 -vocab_path ./tokenizer/korean_32000.model -model_path ./model/ -accum_count 4 -log_file ./log.txt -inter_layer 6,7 -inter_heads 8 -hier -world_size 1 -visible_gpus 0 -gpu_rank 0 -test_from ./model/model.pt
```

## Reference

from [nlpyang/hiersumm](https://github.com/nlpyang/hiersumm).
