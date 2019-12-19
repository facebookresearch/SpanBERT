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

class ParagraphInfo(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary


    def get_word_piece_map(self, sentence):
        return [self.dictionary.is_start_word(i) for i in sentence]

    def get_word_at_k(self, sentence, left, right, k, word_piece_map=None):
        num_words = 0
        while num_words < k and right < len(sentence):
            # complete current word
            left = right
            right = self.get_word_end(sentence, right, word_piece_map)
            num_words += 1
        return left, right

    def get_word_start(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        left  = anchor
        while left > 0 and word_piece_map[left] == False:
            left -= 1
        return left
    # word end is next word start
    def get_word_end(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        right = anchor + 1
        while right < len(sentence) and word_piece_map[right] == False:
            right += 1
        return right

class MaskingScheme:
    def __init__(self, args):
        self.args = args
        self.mask_ratio = getattr(self.args, 'mask_ratio', None)

    def mask(tokens, tagmap=None):
        pass

class BertRandomMaskingScheme(MaskingScheme):
    def __init__(self, args, tokens, pad, mask_id):
        super().__init__(args)
        self.pad = pad
        self.tokens = tokens
        self.mask_id = mask_id

    def mask(self, sentence, tagmap=None):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = np.random.choice(sent_length, mask_num, replace=False)
        return bert_masking(sentence, mask, self.tokens, self.pad, self.mask_id)


class PairWithSpanMaskingScheme(MaskingScheme):
    def __init__(self, args, tokens, pad, mask_id, paragraph_info): 
        super().__init__(args)
        self.args = args
        self.max_pair_targets = args.max_pair_targets
        self.lower = args.span_lower
        self.upper = args.span_upper
        self.pad = pad
        self.mask_id = mask_id
        self.tokens = tokens
        self.paragraph_info = paragraph_info
        self.lens = list(range(self.lower, self.upper + 1))
        self.p = args.geometric_p
        self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        print(self.len_distrib, self.lens)


    def mask(self, sentence, tagmap=None):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            tagged_indices = None
            if tagmap is not None:
                tagged_indices = [max(0, i - np.random.randint(0, span_len)) for i in range(tagmap.length()) if tagmap[i]]
                tagged_indices += [np.random.choice(sent_length)] * int(len(tagged_indices) == 0)
            anchor  = np.random.choice(sent_length) if np.random.rand() >= self.args.tagged_anchor_prob else np.random.choice(tagged_indices)
            if anchor in mask:
                continue
            # find word start, end
            left1, right1 = self.paragraph_info.get_word_start(sentence, anchor, word_piece_map), self.paragraph_info.get_word_end(sentence, anchor, word_piece_map)
            spans.append([left1, left1])
            for i in range(left1, right1):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
            num_words = 1
            right2 = right1
            while num_words < span_len and right2 < len(sentence) and len(mask) < mask_num:
                # complete current word
                left2 = right2
                right2 = self.paragraph_info.get_word_end(sentence, right2, word_piece_map)
                num_words += 1
                for i in range(left2, right2):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
        sentence, target, pair_targets = span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.args.replacement_method, endpoints=self.args.endpoints)
        if self.args.return_only_spans:
            pair_targets = None
        return sentence, target, pair_targets

class NERSpanMaskingScheme(MaskingScheme):
    def __init__(self, args, tokens, pad, mask_id, paragraph_info): 
        super().__init__(args)
        self.args = args
        self.max_pair_targets = args.max_pair_targets
        self.lower = args.span_lower
        self.upper = args.span_upper
        self.pad = pad
        self.mask_id = mask_id
        self.tokens = tokens
        self.paragraph_info = paragraph_info
        self.lens = list(range(self.lower, self.upper + 1))
        self.p = args.geometric_p
        self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        print(self.len_distrib, self.lens)

    def mask_random_span(self, sentence, mask_num, word_piece_map, spans, mask, span_len, anchor):
        # find word start, end
        left1, right1 = self.paragraph_info.get_word_start(sentence, anchor, word_piece_map), self.paragraph_info.get_word_end(sentence, anchor, word_piece_map)
        spans.append([left1, left1])
        for i in range(left1, right1):
            if len(mask) >= mask_num:
                break
            mask.add(i)
            spans[-1][-1] = i
        num_words = 1
        right2 = right1
        while num_words < span_len and right2 < len(sentence) and len(mask) < mask_num:
            # complete current word
            left2 = right2
            right2 = self.paragraph_info.get_word_end(sentence, right2, word_piece_map)
            num_words += 1
            for i in range(left2, right2):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i

    def mask_entity(self, sentence, mask_num, word_piece_map, spans, mask, entity_spans):
        if len(entity_spans) > 0:
            entity_span = entity_spans[np.random.choice(range(len(entity_spans)))]
            spans.append([entity_span[0], entity_span[0]])
            for idx in range(entity_span[0], entity_span[1] + 1):
                if len(mask) >= mask_num:
                    break
                spans[-1][-1] = idx
                mask.add(idx)


    def mask(self, sentence, entity_map=None):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
        # get entity spans
        entity_spans, spans = [], []
        new_entity = True
        for i in range(entity_map.length()):
            if entity_map[i] and new_entity:
                entity_spans.append([i, i])
                new_entity = False
            elif entity_map[i] and not new_entity:
                entity_spans[-1][-1] = i
            else:
                new_entity = True
        while len(mask) < mask_num:
            if np.random.random() <= self.args.ner_masking_prob:
                self.mask_entity(sentence, mask_num, word_piece_map, spans, mask, entity_spans)
            else:
                span_len = np.random.choice(self.lens, p=self.len_distrib)
                anchor  = np.random.choice(sent_length)
                if anchor in mask:
                    continue
                self.mask_random_span(sentence, mask_num, word_piece_map, spans, mask, span_len, anchor)
        sentence, target, pair_targets = span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.args.replacement_method, endpoints=self.args.endpoints)
        if self.args.return_only_spans:
            pair_targets = None
        return sentence, target, pair_targets


def bert_masking(sentence, mask, tokens, pad, mask_id):
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.copy(sentence)
    mask = set(mask)
    for i in range(sent_length):
        if i in mask:
            rand = np.random.random()
            if rand < 0.8:
                sentence[i] = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence[i] = np.random.choice(tokens)
        else:
            target[i] = pad
    return sentence, target, None

def span_masking(sentence, spans, tokens, pad, mask_id, pad_len, mask, replacement='word_piece', endpoints='external'):
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.full(sent_length, pad)
    pair_targets = []
    spans = merge_intervals(spans)
    assert len(mask) == sum([e - s + 1 for s,e in spans])
    # print(list(enumerate(sentence)))
    for start, end in spans:
        lower_limit = 0 if endpoints == 'external' else -1
        upper_limit = sent_length - 1 if endpoints == 'external' else sent_length
        if start > lower_limit and end < upper_limit:
            if endpoints == 'external':
                pair_targets += [[start - 1, end + 1]]
            else:
                pair_targets += [[start, end]]
            pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
        rand = np.random.random()
        for i in range(start, end + 1):
            assert i in mask
            target[i] = sentence[i]
            if replacement == 'word_piece':
                rand = np.random.random()
            if rand < 0.8:
                sentence[i] = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence[i] = np.random.choice(tokens)
    pair_targets = pad_to_len(pair_targets, pad, pad_len + 2)
    # if pair_targets is None:
    return sentence, target, pair_targets

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x : x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged

def pad_to_max(pair_targets, pad):
    max_pair_target_len = max([len(pair_tgt) for pair_tgt in pair_targets])
    for pair_tgt in pair_targets:
        this_len = len(pair_tgt)
        for i in range(this_len, max_pair_target_len):
            pair_tgt.append(pad)
    return pair_targets

def pad_to_len(pair_targets, pad, max_pair_target_len):
    for i in range(len(pair_targets)):
        pair_targets[i] = pair_targets[i][:max_pair_target_len]
        this_len = len(pair_targets[i])
        for j in range(max_pair_target_len - this_len):
            pair_targets[i].append(pad)
    return pair_targets

