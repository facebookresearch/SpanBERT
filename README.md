# SpanBERT
This repository contains code and models for the paper: [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529).

## Requirements

## Pre-trained Models
We release both base and large *cased* models for SpanBERT. The base & large models have the same model configuration as [BERT](https://github.com/google-research/bert) but they differ in
both the masking scheme and the training objectives (see our paper for more details).

* [SpanBERT (base & cased)](https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz): 12-layer, 768-hidden, 12-heads , 110M parameters
* [SpanBERT (large & cased)](https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz): 24-layer, 1024-hidden, 16-heads, 340M parameters

These models have the same format as the [HuggingFace BERT models](https://github.com/huggingface/pytorch-transformers), so you can easily replace them with our SpanBET models. If you would like to use our [fine-tuning code](#fine-tuning), the model paths are already hard-coded in the code :)


<!-- |                   | SQuAD 1.1     | SQuAD 2.0  | Coref   | TACRED | NewsQA | TriviaQA | SearchQA | HotpotQA | NaturalQ |
| ----------------------  | ------------- | ---------  | ------- | ------ | ------ | -------- | -------- | ------- | ------ |
|                         | F1            | F1         | avg. F1 |  F1    | F1     | F1       | F1       | F1 | F1 |
| BERT (base)             | 88.5*         | 76.5*      | 73.1    |  67.7  | 65.4   | 74.2	| 79.8 | 77.0	| 76.5 |
| SpanBERT (base)         | 92.4*         | 83.6*      | 77.4    |  68.2  | 70.4   | 79.2	| 82.9 | 81.0	| 80.6 |
| BERT (large)            | 91.3          | 83.3       | 77.1    |  66.4  | 68.8   | 77.5 | 81.7 | 78.3 | 79.9 |
| SpanBERT (large)        | 94.6          | 88.7       | 79.6    |  70.8  | 73.6   | 83.6 | 84.8 | 83.0 | 82.5 | -->

|                   | SQuAD 1.1     | SQuAD 2.0  | Coref   | TACRED |
| ----------------------  | ------------- | ---------  | ------- | ------ |
|                         | F1            | F1         | avg. F1 |  F1    |
| BERT (base)             | 88.5*         | 76.5*      | 73.1    |  67.7  |
| SpanBERT (base)         | 92.4*         | 83.6*      | 77.4    |  68.2  |
| BERT (large)            | 91.3          | 83.3       | 77.1    |  66.4  |
| SpanBERT (large)        | 94.6          | 88.7       | 79.6    |  70.8  |


Note: The numbers marked as * are evaluated on the development sets because we didn't submit those models to the official SQuAD leaderboard. All the other numbers are test numbers.


## Fine-tuning

### SQuAD 1.1

```bash
python code/run_squad.py \
  --do_train \
  --do_eval \
  --model spanbert-base-cased \
  --train_file train-v1.1.json \
  --dev_file dev-v1.1.json \
  --train_batch_size 32 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir squad_output \
  --fp16
```

### SQuAD 2.0

### Coref

### TACRED

### MRQA (NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions)

### GLUE

## Available models (QA, Coreference, Relation Extraction)

If you are interested in using our fine-tuned models for downstream tasks (QA, coreference, relation extraction) directly, we also provide the following models:




## Citation
```
  @article{joshi2019spanbert,
      title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},
      author={Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
      journal={arXiv preprint arXiv:1907.10529},
      year={2019}
    }
```

## License
TODO

## Contact
If you have any questions, please contact Mandar Joshi `<mandar90@cs.washington.edu>` or Danqi Chen `<danqic@cs.princeton.edu>` or create a Github issue.
