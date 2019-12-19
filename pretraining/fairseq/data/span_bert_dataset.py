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

class BlockPairDataset(torch.utils.data.Dataset):
    """Break a 1d tensor of tokens into sentence pair blocks for next sentence
       prediction as well as masked language model.
       High-level logics are:
       1. break input tensor to tensor blocks
       2. pair the blocks with 50% next sentence and 50% random sentence
       3. return paired blocks as well as related segment labels
    Args:
        tokens: 1d tensor of tokens to break into blocks
        block_size: maximum block size
        pad: pad index
        eos: eos index
        cls: cls index
        mask: mask index
        sep: sep index to separate blocks
    """

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
        if break_mode == "sentence":
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
            curr = 0
            for sz in sizes:
                if sz == 0:
                    continue
                self.block_indices.append((curr, curr + sz))
                curr += sz
            max_num_tokens = block_size - 3 # Account for [CLS], [SEP], [SEP]
            self.sent_pairs = []
            self.sizes = []
            target_seq_length = max_num_tokens
            if np.random.random() < short_seq_prob:
                target_seq_length = np.random.randint(2, max_num_tokens)
            current_chunk = []
            current_length = 0
            curr = 0
            while curr < len(self.block_indices):
                sent = self.block_indices[curr]
                current_chunk.append(sent)
                current_length = current_chunk[-1][1] - current_chunk[0][0]
                if curr == len(self.block_indices) - 1 or current_length >= target_seq_length:
                    if current_chunk:
                        a_end = 1
                        if len(current_chunk) > 2:
                            a_end = np.random.randint(1, len(current_chunk) - 1)
                        sent_a = current_chunk[:a_end]
                        sent_a = (sent_a[0][0], sent_a[-1][1])
                        next_sent_label = (
                            1 if np.random.rand() > 0.5 else 0
                        )
                        if len(current_chunk) == 1 or next_sent_label:
                            target_b_length = target_seq_length - (sent_a[1] - sent_a[0])
                            random_start = np.random.randint(0, len(self.block_indices) - len(current_chunk))
                            # avoid current chunks
                            # we do this just because we don't have document level segumentation
                            random_start = (
                                random_start + len(current_chunk)
                                if self.block_indices[random_start][1] > current_chunk[0][0]
                                else random_start
                            )
                            sent_b = []
                            for j in range(random_start, len(self.block_indices)):
                                sent_b = (
                                    (sent_b[0], self.block_indices[j][1])
                                    if sent_b else self.block_indices[j]
                                )
                                if self.block_indices[j][0] == current_chunk[0][0]:
                                    break
                                # length constraint
                                if sent_b[1] - sent_b[0] >= target_b_length:
                                    break
                            num_unused_segments = len(current_chunk) - a_end
                            curr -= num_unused_segments
                            next_sent_label = 1
                        else:
                            sent_b = current_chunk[a_end:]
                            sent_b = (sent_b[0][0], sent_b[-1][1])
                        sent_a, sent_b = self._truncate_sentences(sent_a, sent_b, max_num_tokens)
                        self.sent_pairs.append((sent_a, sent_b, next_sent_label))
                        if sent_a[0] >= sent_a[1] or sent_b[0] >= sent_b[1]:
                            print(sent_a, sent_b)
                        self.sizes.append(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0])
                    current_chunk = []
                curr += 1
        elif break_mode == "doc":
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
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
            max_num_tokens = block_size - 3 # Account for [CLS], [SEP], [SEP]
            self.sent_pairs = []
            self.sizes = []
            for doc_id, doc in enumerate(self.block_indices):
                current_chunk = []
                current_length = 0
                curr = 0
                target_seq_length = max_num_tokens
                short = False
                if np.random.random() < short_seq_prob:
                    short = True
                    target_seq_length = np.random.randint(2, max_num_tokens)
                while curr < len(doc):
                    sent = doc[curr]
                    current_chunk.append(sent)
                    current_length = current_chunk[-1][1] - current_chunk[0][0]
                    if curr == len(doc) - 1 or current_length >= target_seq_length:
                        if current_chunk:
                            a_end = 1
                            if len(current_chunk) > 2:
                                a_end = np.random.randint(1, len(current_chunk) - 1)
                            sent_a = current_chunk[:a_end]
                            sent_a = (sent_a[0][0], sent_a[-1][1])
                            next_sent_label = (
                                1 if np.random.rand() > 0.5 else 0
                            )
                            if len(current_chunk) == 1 or next_sent_label:
                                next_sent_label = 1
                                target_b_length = target_seq_length - (sent_a[1] - sent_a[0])
                                for _ in range(10):
                                    rand_doc_id = np.random.randint(0, len(self.block_indices) - 1)
                                    if rand_doc_id != doc_id:
                                        break
                                random_doc = self.block_indices[rand_doc_id]
                                random_start = np.random.randint(0, len(random_doc))
                                sent_b = []
                                for j in range(random_start, len(random_doc)):
                                    sent_b = (
                                        (sent_b[0], random_doc[j][1])
                                        if sent_b else random_doc[j]
                                    )
                                    if sent_b[1] - sent_b[0] >= target_b_length:
                                        break
                                num_unused_segments = len(current_chunk) - a_end
                                curr -= num_unused_segments
                            else:
                                next_sent_label = 0
                                sent_b = current_chunk[a_end:]
                                sent_b = (sent_b[0][0], sent_b[-1][1])
                            sent_a, sent_b = self._truncate_sentences(sent_a, sent_b, max_num_tokens)
                            self.sent_pairs.append((sent_a, sent_b, next_sent_label))
                            if sent_a[0] >= sent_a[1] or sent_b[0] >= sent_b[1]:
                                print(sent_a, sent_b)
                            self.sizes.append(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0])
                            #print(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0], curr == len(doc) - 1, short)
                        current_chunk = []
                    curr += 1
        else:
            block_size -= 3
            block_size //= 2  # each block should have half of the block size since we are constructing block pair
            length = math.ceil(len(tokens) / block_size)

            def block_at(i):
                start = i * block_size
                end = min(start + block_size, len(tokens))
                return (start, end)

            self.block_indices = [block_at(i) for i in range(length)]

            self.sizes = np.array(
                # 2 block lengths + 1 cls token + 2 sep tokens
                # note: this is not accurate and larger than pairs including last block
                [block_size * 2 + 3] * len(self.block_indices)
            )

    def _truncate_sentences(self, sent_a, sent_b, max_num_tokens):
        while True:
            total_length = sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0]
            if total_length <= max_num_tokens:
                return sent_a, sent_b

            if sent_a[1] - sent_a[0] > sent_b[1] - sent_b[0]:
                sent_a = (
                    (sent_a[0]+1, sent_a[1])
                    if np.random.rand() < 0.5
                    else (sent_a[0], sent_a[1] - 1)
                )
            else:
                sent_b = (
                    (sent_b[0]+1, sent_b[1])
                    if np.random.rand() < 0.5
                    else (sent_b[0], sent_b[1] - 1)
                )

    def _rand_block_index(self, i):
        """select a random block index which is not given block or next
           block
        """
        idx = np.random.randint(len(self.block_indices) - 3)
        return idx if idx < i else idx + 2


    def __getitem__(self, index):
        if self.break_mode == "sentence" or self.break_mode == "doc":
            block1, block2, next_sent_label = self.sent_pairs[index]
        else:
            next_sent_label = (
                1 if np.random.rand() > 0.5 else 0
            )
            block1 = self.block_indices[index]
            if next_sent_label:
                block2 = self.block_indices[self._rand_block_index(index)]
            elif index == len(self.block_indices) - 1:
                next_sent_label = 1
                block2 = self.block_indices[self._rand_block_index(index)]
            else:
                block2 = self.block_indices[index+1]

        return block1, block2, next_sent_label

    def __len__(self):
        return len(self.sizes)


class SpanBertDataset(FairseqDataset):
    """
    A wrapper around BlockPairDataset for BERT data.
    Args:
        dataset (BlockPairDataset): dataset to wrap
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


    def _mask_block(self, sentence, tagmap):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        masking_scheme = random.choice(self.masking_schemes)
        return masking_scheme.mask(sentence, tagmap)

    def _get_offsets(self, size):
        probs = np.random.random(size=size)
        return [np.random.randint(-pos, size - pos) for pos, p in enumerate(probs)]

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            (
                block1, block2, next_sent_label
            ) = self.dataset[index]
        tagmap_1, tagmap_2 = (self.dataset.tag_map[block1[0]:block1[1]], self.dataset.tag_map[block2[0]:block2[1]]) if self.dataset.tag_map is not None else (None, None)
        masked_blk1, masked_tgt1, pair_targets_1 = self._mask_block(self.dataset.tokens[block1[0]:block1[1]], tagmap_1)
        masked_blk2, masked_tgt2, pair_targets_2 = self._mask_block(self.dataset.tokens[block2[0]:block2[1]], tagmap_2)

        item1 = np.concatenate(
            [
                [self.vocab.cls()],
                masked_blk1,
                [self.vocab.sep()],
            ]

        )
        item2 = np.concatenate(
            [
                masked_blk2,
                [self.vocab.sep()],
            ]

        )
        if self.dataset.tag_map is not None:
            tagmap_1, tagmap_2 = self.dataset.tag_map[block1[0]:block1[1]], self.dataset.tag_map[block2[0]:block2[1]]

        target1 = np.concatenate([[self.vocab.pad()], masked_tgt1, [self.vocab.pad()]])
        target2 = np.concatenate([masked_tgt2, [self.vocab.pad()]])

        seg1 = np.zeros((block1[1] - block1[0]) + 2)  # block + 1 sep + 1 cls
        seg2 = np.ones((block2[1] - block2[0]) + 1)  # block + 1 sep

        item = np.concatenate([item1, item2])

        seg = np.concatenate([seg1, seg2])
        target = np.concatenate([target1, target2])
        pair_targets = None
        if pair_targets_1 is not None and pair_targets_2 is not None and len(pair_targets_1) + len(pair_targets_2) > 0:
            assert pair_targets_2 is not None
            # dummy = [[0 for i in range(self.args.max_pair_targets + 2)]]
            pair_targets_1 = [[(x+1) if i < 2 else x for i, x in enumerate(pair_tgt)] for pair_tgt in pair_targets_1]
            pair_targets_2 = [[(x+item1.shape[0]) if i < 2 else x for i, x in enumerate(pair_tgt)] for pair_tgt in pair_targets_2]
            #pair_targets = dummy + pair_targets_1 + pair_targets_2 # pad_to_max(pair_targets_1 + pair_targets_2, self.vocab.pad())
            pair_targets = pair_targets_1 + pair_targets_2 # pad_to_max(pair_targets_1 + pair_targets_2, self.vocab.pad())
            pair_targets = torch.from_numpy(np.array(pair_targets)).long()
        else:
            pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        return {
            'id': index,
            'source': torch.from_numpy(item).long(),
            'segment_labels': torch.from_numpy(seg).long(),
            'lm_target': torch.from_numpy(target).long(),
            'sentence_target': next_sent_label,
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
            'sentence_target': torch.LongTensor([s['sentence_target'] for s in samples]),
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
        sentence_target = 0
        bsz = num_tokens // tgt_len

        return self.collater([
            {
                'id': i,
                'source': source,
                'segment_labels': segment_labels,
                'lm_target': lm_target,
                'sentence_target': sentence_target,
                'pair_targets': pair_targets,
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

