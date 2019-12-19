# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from tqdm import tqdm
import numpy as np
import torch
from math import sqrt, log
from . import data_utils, FairseqDataset
from itertools import chain
import json
import random
from collections import defaultdict
from fairseq.data.masking import ParagraphInfo, BertRandomMaskingScheme, PairWithSpanMaskingScheme, NERSpanMaskingScheme

class BlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokens,
        sizes,
        block_size,
        pad,
        cls,
        mask,
        sep,
        break_mode="doc",
        short_seq_prob=0.1,
        tag_map=None
    ):
        super().__init__()
        self.tokens = tokens
        self.total_size = len(tokens)
        self.pad = pad
        self.cls = cls
        self.mask = mask
        self.sep = sep
        self.block_indices = []
        self.break_mode = break_mode
        self.tag_map = tag_map
        if tag_map is not None:
            print('Len of tag map: {} len of corpus: {}'.format(tag_map.length(), len(tokens)))

        assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
        max_num_tokens = block_size - 2
        self.sents = []
        self.sizes = []

        if break_mode == "sentence":
            curr = 0
            for sz in sizes:
                if sz == 0:
                    continue
                self.block_indices.append((curr, curr + sz))
                curr += sz
            for curr in range(len(self.block_indices)):
                sent = self.block_indices[curr]
                if sent[1] - sent[0] <= max_num_tokens:
                    self.sents.append(sent)
                    self.sizes.append(sent[1] - sent[0] + 2)
                    if len(self.sents) <= 5:
                        print("Sentence: %s (sz = %d)" % (self.sents[-1], self.sizes[-1]))
        elif break_mode == "doc":
            curr = 0
            cur_doc = []
            for sz in sizes:
                if sz == 0:
                    if len(cur_doc) == 0: continue
                    self.block_indices.append(cur_doc)
                    cur_doc = []
                else:
                    cur_doc.append((curr, curr + sz))
                curr += sz
            for doc in self.block_indices:
                current_chunk = []
                curr = 0
                while curr < len(doc):
                    sent = doc[curr]
                    if sent[1] - sent[0] <= max_num_tokens:
                        current_chunk.append(sent)
                        current_length = current_chunk[-1][1] - current_chunk[0][0]
                        if curr == len(doc) - 1 or current_length > max_num_tokens:
                            if current_length > max_num_tokens:
                                current_chunk = current_chunk[:-1]
                                curr -= 1
                            if len(current_chunk) > 0:
                                sent = (current_chunk[0][0], current_chunk[-1][1])
                                self.sents.append(sent)
                                self.sizes.append(sent[1] - sent[0] + 2)
                                if len(self.sents) <= 5:
                                    print("Sentence: %s (sz = %d)" % (self.sents[-1], self.sizes[-1]))
                            current_chunk = []
                    curr += 1

        else:
            raise ValueError("break_mode = %s not supported." % self.break_mode)

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sizes)


class NoNSPSpanBertDataset(FairseqDataset):
    """
    A wrapper around BlockDataset for BERT data.
    Args:
        dataset (BlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, sizes, vocab, shuffle, seed,
        		args=None):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.masking_schemes = []
        self.paragraph_info = ParagraphInfo(vocab)
        self.args = args
        for scheme_str in json.loads(args.schemes):
            if scheme_str == 'random':
                scheme = BertRandomMaskingScheme(args, self.dataset.tokens, self.dataset.pad, self.dataset.mask)
            elif scheme_str == 'pair_span':
                scheme = PairWithSpanMaskingScheme(args, self.dataset.tokens, self.dataset.pad, self.dataset.mask, self.paragraph_info)
            elif scheme_str == 'ner_span':
                scheme = NERSpanMaskingScheme(args, self.dataset.tokens, self.dataset.pad, self.dataset.mask, self.paragraph_info)
            else:
                raise NotImplementedError()
            self.masking_schemes.append(scheme)

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            block = self.dataset[index]

        tagmap = self.dataset.tag_map[block[0]:block[1]] if self.dataset.tag_map is not None else None
        masked_block, masked_tgt, pair_targets = \
            self._mask_block(self.dataset.tokens[block[0]:block[1]], tagmap)

        item = np.concatenate(
            [
                [self.vocab.cls()],
                masked_block,
                [self.vocab.sep()],
            ]
        )
        target = np.concatenate([[self.vocab.pad()], masked_tgt, [self.vocab.pad()]])
        seg = np.zeros(block[1] - block[0] + 2)
        if pair_targets is not None and  len(pair_targets) > 0:
            # dummy = [[0 for i in range(self.args.max_pair_targets + 2)]]
            # add 1 to the first two since they are input indices. Rest are targets.
            pair_targets = [[(x+1) if i < 2 else x for i, x in enumerate(pair_tgt)] for pair_tgt in pair_targets]
            # pair_targets = dummy + pair_targets
            pair_targets = torch.from_numpy(np.array(pair_targets)).long()
        else:
            pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        return {
            'id': index,
            'source': torch.from_numpy(item).long(),
            'segment_labels': torch.from_numpy(seg).long(),
            'lm_target': torch.from_numpy(target).long(),
            'pair_targets': pair_targets,
        }

    def __len__(self):
        return len(self.dataset)

    def _collate(self, samples, pad_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        def merge_2d(key):
            return data_utils.collate_2d(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        pair_targets = merge_2d('pair_targets')

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': merge('source'),
                'segment_labels': merge('segment_labels'),
                'pairs': pair_targets[:, :, :2]
            },
            'lm_target': merge('lm_target'),
            'nsentences': samples[0]['source'].size(0),
            'pair_targets': pair_targets[:, :, 2:]
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return self._collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=12):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        segment_labels = torch.zeros(tgt_len, dtype=torch.long)
        pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        lm_target = source
        bsz = num_tokens // tgt_len

        return self.collater([
            {
                'id': i,
                'source': source,
                'segment_labels': segment_labels,
                'lm_target': lm_target,
                'pair_targets': pair_targets
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def _mask_block(self, sentence, tagmap):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        masking_scheme = random.choice(self.masking_schemes)
        block = masking_scheme.mask(sentence, tagmap)
        return block
