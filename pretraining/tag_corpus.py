import sys
import os
from tqdm import tqdm
import multiprocessing
sys.path.append('../pytorch-pretrained-BERT')
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer
import spacy

def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc

def get_chunks(fpath, chunk_size):
    f = open(fpath)
    chunk = []
    for line in f:
        chunk.append(line.strip())
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    yield chunk

def np_chunk(cased_lines, uncased_lines, nlp, tokenizer, basic_tokenizer, worker_id, batch_offset):
    tagged_sents = []
    for cased_line, uncased_line in zip(cased_lines, uncased_lines):
        tokens = basic_tokenizer.tokenize(cased_line)
        uncased_subtokens = uncased_line.split(' ') if uncased_line else []
        doc = nlp.tokenizer.tokens_from_list(tokens)
        doc = prevent_sentence_boundary_detection(doc)
        doc = nlp.tagger(doc)
        doc = nlp.parser(doc)
        split_tokens, tags = [], []
        np_indices = sorted((np.start, np.end) for np in doc.noun_chunks)
        curr = 0
        for i, token in enumerate(doc):
            while curr < len(np_indices) and np_indices[curr][1] - 1 < i:
                curr += 1
            subtokens = tokenizer.tokenize(token.text)
            split_tokens += subtokens
            tags += ['NP' if (curr < len(np_indices) and i >= np_indices[curr][0] and i < np_indices[curr][1]) else 'NOTAG'] * len(subtokens)
        if len(uncased_subtokens) == len(split_tokens):
            tagged_sents.append(' '.join([token + ' ' + tag for token, tag in zip(uncased_subtokens, tags)]))
        else:
            tagged_sents.append(' '.join([token + ' ' + 'NOTAG' for token in uncased_subtokens]))
    return worker_id, tagged_sents, batch_offset

def ner_tag(cased_lines, uncased_lines, nlp, tokenizer, basic_tokenizer, worker_id, batch_offset):
    tagged_sents = []
    for cased_line, uncased_line in zip(cased_lines, uncased_lines):
        tokens = basic_tokenizer.tokenize(cased_line)
        uncased_subtokens = uncased_line.split(' ') if uncased_line else []
        doc = nlp.tokenizer.tokens_from_list(tokens)
        doc = prevent_sentence_boundary_detection(doc)
        nlp.entity(doc)
        split_tokens, tags = [], []
        for token in doc:
            subtokens = tokenizer.tokenize(token.text)
            split_tokens += subtokens
            tags += [(token.ent_type_ if token.ent_type_ != '' else 'NOTAG')] * len(subtokens)
        if len(uncased_subtokens) == len(split_tokens):
            tagged_sents.append(' '.join([token + ' ' + tag for token, tag in zip(uncased_subtokens, tags)]))
        else:
            tagged_sents.append(' '.join([token + ' ' + 'NOTAG' for token in uncased_subtokens]))
    return worker_id, tagged_sents, batch_offset

def process(cased_file, uncased_file, output_file, bert_model_type='bert-base-cased', tag_type='ne', total=180378072, chunk_size=1000000, workers=40):
    results = list(range(workers))
    nlp = spacy.load('en', vectors=False)
    tokenizer = BertTokenizer.from_pretrained(bert_model_type)
    basic_tokenizer = BasicTokenizer(do_lower_case=False)
    fout = open(output_file, 'w')
    offset = 0
    tagging_fn = np_chunk if tag_type == 'np' else ner_tag
    def merge_fn(result):
        worker_id, tokenized, batch_offset = result
        results[worker_id] = tokenized, batch_offset
    for uncased_lines, cased_lines in tqdm(zip(get_chunks(uncased_file, chunk_size), get_chunks(cased_file, chunk_size)), total=total//chunk_size):
        pool = multiprocessing.Pool()
        assert len(cased_lines) == len(uncased_lines)
        size = (len(cased_lines) // workers) if len(cased_lines) % workers == 0 else ( 1 + (len(cased_lines) // workers))
        for i in range(workers):
            start = i * size
            pool.apply_async(tagging_fn, args = (cased_lines[start:start+size], uncased_lines[start: start + size], nlp, tokenizer, basic_tokenizer, i, start), callback = merge_fn)
        pool.close()
        pool.join()
        for lines, batch_offset in results:
            for i, line in enumerate(lines):
                fout.write(str(offset + batch_offset + i) + ' ' + line + '\n')
        offset += len(cased_lines)


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2], sys.argv[3])
