# Preprocessing
* Tokenization
This step assumes that the corpus is formatted as one-sentence-per-line with line breaks indicating the end of a document.
```
python bpe_tokenize.py /path/to/cased_corpus_file /path/to/output_file
```

* Fairseq Preprocessing
```
python preprocess.py --only-source --trainpref /path/to/train.txt --validpref path/to/valid.txt --srcdict /path/to/dict.txt --destdir /path/to/destination_dir --padding-factor 1 --workers 48
```

Make sure dict.txt contains two columns. The first is the word peice and the second is the dummy frequency in descending order.

# Pre-training
Use this command to start pretraining. The options coresspond to the SpanBERT configuration mentioned in the paper.
```
python train.py /path/to/preprocessed_data --total-num-update 2400000 --max-update 2400000  --save-interval 1 --arch cased_bert_pair_large --task  span_bert --optimizer adam --lr-scheduler polynomial_decay --lr 0.0001 --min-lr 1e-09  --criterion span_bert_loss  --max-tokens 4096 --tokens-per-sample 512 --weight-decay 0.01  --skip-invalid-size-inputs-valid-test --log-format json --log-interval 2000 --save-interval-updates 50000 --keep-interval-updates 50000 --update-freq 1 --seed 1 --save-dir /path/to/checkpoint_dir --fp16 --warmup-updates 10000 --schemes [\"pair_span\"] --distributed-port 12580 --distributed-world-size 32 --span-lower 1 --span-upper 10 --validate-interval 1  --clip-norm 1.0 --geometric-p 0.2 --adam-eps 1e-8 --short-seq-prob 0.0 --replacement-method span --clamp-attention --no-nsp --pair-loss-weight 1.0 --max-pair-targets 15 --pair-positional-embedding-size 200 --endpoints external
```

# Important Files
* `fairseq/tasks/span_bert.py`: Main task file which also contains the all the task-specific options.
* `fairseq/data/no_nsp_span_bert_dataset.py`: This is where the data preprocessing happens.
* `fairseq/data/masking.py`: All the masking schemes are defined here. These are called from the dataset files above.
* `fairseq/criterions` -- `span_bert_loss`: The losses are defined here. **Make sure `--no_-nsp` is set to true when using the no_-nsp losses**
* `fairseq/models/pair_bert.py`: Transformer model.


