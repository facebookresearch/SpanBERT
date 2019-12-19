# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,
    data_utils,
)

from fairseq.data.span_bert_dataset import (
    BlockPairDataset,
    SpanBertDataset,
)
from fairseq.data.no_nsp_span_bert_dataset import (
    BlockDataset,
    NoNSPSpanBertDataset,
)

from . import FairseqTask, register_task
from fairseq.data.masking import ParagraphInfo
from bitarray import bitarray

class BertDictionary(Dictionary):
    """Dictionary for BERT tasks
        extended from Dictionary by adding support for cls as well as mask symbols"""
    def __init__(
        self,
        pad='[PAD]',
        unk='[UNK]',
        cls='[CLS]',
        mask='[MASK]',
        sep='[SEP]'
    ):
        super().__init__(pad, unk)
        (
            self.cls_word,
            self.mask_word,
            self.sep_word,
        ) = cls, mask, sep
        self.is_start = None
        self.nspecial = len(self.symbols)

    def class_positive(self):
        return self.cls()

    def cls(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.cls_word)
        return idx

    def mask(self):
        """Helper to get index of mask symbol"""
        idx = self.add_symbol(self.mask_word)
        return idx

    def sep(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.sep_word)
        return idx

    def is_start_word(self, idx):
        if self.is_start is None:
            self.is_start = [not self.symbols[i].startswith('##') for i in range(len(self))]
        return self.is_start[idx]


@register_task('span_bert')
class SpanBertTask(FairseqTask):
    """
    Train BERT model.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments'
                                 ' per sample for BERT dataset')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--break-mode', default="doc", type=str, help='mode for breaking sentence')
        parser.add_argument('--schemes', default='["random"]', type=str, help='list of masking schemes')
        parser.add_argument('--span-lower', default=1, type=int, help='lower bound on the number of words in a span')
        parser.add_argument('--span-upper', default=10, type=int, help='upper bound on the number of words in a span')
        parser.add_argument('--max-pair-targets', default=20, type=int, help='max word pieces b/w a pair')
        parser.add_argument('--mask-ratio', default=0.15, type=float, help='proportion of words to be masked')
        parser.add_argument('--geometric-p', default=0.3, type=float, help='p for the geometric distribution used in span masking. -1 is uniform')
        parser.add_argument('--pair-loss-weight', default=0.0, type=float, help='weight for pair2/SBO loss')
        parser.add_argument('--tagged-anchor-prob', default=0.0, type=float, help='prob of selecting an anchor according to the tag bitmap')
        parser.add_argument('--short-seq-prob', default=0.1, type=float)
        parser.add_argument('--pair-target-layer', default=-1, type=int)
        parser.add_argument('--pair-positional-embedding-size', default=200, type=int)
        parser.add_argument('--ner-masking-prob', default=0.5, type=float)
        parser.add_argument('--replacement-method', default='word_piece')
        parser.add_argument('--return-only-spans', default=False, action='store_true')
        parser.add_argument('--shuffle-instance', default=False, action='store_true')
        parser.add_argument('--no-nsp', default=False, action='store_true')
        parser.add_argument('--endpoints', default='external', type=str)
        parser.add_argument('--skip-validation', default=False, action='store_true')
        parser.add_argument('--tag-bitmap-file-prefix', default=None, help='file containing bitmap of verb tokens')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        args.vocab_size = len(dictionary)
        self.seed = args.seed
        self.no_nsp = args.no_nsp
        self.short_seq_prob = args.short_seq_prob

    @property
    def target_dictionary(self):
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(self.args.data, split_k)

            if self.args.raw_text and IndexedRawTextDataset.exists(path):
                ds = IndexedRawTextDataset(path, self.dictionary)
                tokens = [t for l in ds.tokens_list for t in l]
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(path):
                ds = IndexedInMemoryDataset(path, fix_lua_indexing=False)
                tokens = ds.buffer
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))
            tag_map = None
            if self.args.tag_bitmap_file_prefix is not None:
                tag_map = bitarray()
                tag_map.fromfile(open(self.args.tag_bitmap_file_prefix + split, 'rb'))
            block_cls = BlockPairDataset if not self.no_nsp else BlockDataset
            with data_utils.numpy_seed(self.seed + k):
                loaded_datasets.append(
                    block_cls(
                        tokens,
                        ds.sizes,
                        self.args.tokens_per_sample,
                        pad=self.dictionary.pad(),
                        cls=self.dictionary.cls(),
                        mask=self.dictionary.mask(),
                        sep=self.dictionary.sep(),
                        break_mode=self.args.break_mode,
                        short_seq_prob=self.short_seq_prob,
                        tag_map=tag_map

                    ))

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])
        dataset_cls = SpanBertDataset if not self.no_nsp else NoNSPSpanBertDataset
        self.datasets[split] = dataset_cls(
            dataset, sizes, self.dictionary,
            shuffle=self.args.shuffle_instance, seed=self.seed, args=self.args
        )
