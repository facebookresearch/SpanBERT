# SpanBERT
This repository contains code and models for the paper: [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529). If you prefer to use Huggingface, please check out this link -- https://huggingface.co/SpanBERT 

## Requirements
## Apex
Please use an earlier commit of Apex - [NVIDIA/apex@4a8c4ac](https://github.com/NVIDIA/apex/commit/4a8c4ac088b6f84a10569ee89db3a938b48922b4)

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
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir squad_output \
  --fp16
```

### SQuAD 2.0

```bash
python code/run_squad.py \
  --do_train \
  --do_eval \
  --model spanbert-base-cased \
  --train_file train-v2.0.json \
  --dev_file dev-v2.0.json \
  --train_batch_size 32 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric best_f1 \
  --output_dir squad2_output \
  --version_2_with_negative \
  --fp16
```

### TACRED

```bash
python code/run_tacred.py \
  --do_train \
  --do_eval \
  --data_dir <TACRED_DATA_DIR> \
  --model spanbert-base-cased \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir tacred_dir \
  --fp16
```

### MRQA (NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions)

```bash
python code/run_mrqa.py \
  --do_train \
  --do_eval \
  --model spanbert-base-cased \
  --train_file TriviaQA-train.jsonl.gz \
  --dev_file TriviaQA-dev.jsonl.gz \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_per_epoch 5 \
  --output_dir triviaqa_dir \
  --fp16
```

### GLUE

```bash
python code/run_glue.py \
   --task_name RTE \
   --model spanbert-base-cased \
   --do_train \
   --do_eval \
   --data_dir <RTE_DATA_DIR> \
   --train_batch_size 32 \
   --eval_batch_size 32 \
   --num_train_epochs 10  \
   --max_seq_length 128 \
   --learning_rate 2e-5 \
   --output_dir RTE_DIR \
   --fp16
```

### Coreference Resolution
Our coreference resolution fine-tuning code is implemented in Tensorflow. Please see [https://github.com/mandarjoshi90/coref](https://github.com/mandarjoshi90/coref) for more details.

## Finetuned Models (SQuAD 1.1/2.0, Relation Extraction, Coreference Resolution)

If you are interested in using our fine-tuned models for downstream tasks, directly, please use the following script.

```
./code/download_finetuned.sh <model_dir> <task>
```
where `<task>` is one of `[squad1, squad2, tacred]`. You can evaluate the models by setting `--do_train` to `false`, `--do_eval` to `true`, and `--output_dir` to `<model_dir>/<task>` in `python code/run_<task>.py`.

For coreference resolution, please refer to this repository -- [https://github.com/mandarjoshi90/coref](https://github.com/mandarjoshi90/coref)




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
SpanBERT is CC-BY-NC 4.0. The license applies to the pre-trained models as well.

## Contact
If you have any questions, please contact Mandar Joshi `<mandar90@cs.washington.edu>` or Danqi Chen `<danqic@cs.princeton.edu>` or create a Github issue.
