import sys
from tqdm import tqdm
import multiprocessing
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

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


def tokenize(cased_lines, tokenizer, basic_tokenizer, worker_id, batch_offset):
    sents = []
    for cased_line in cased_lines:
        tokens = basic_tokenizer.tokenize(cased_line)
        split_tokens = []
        for token in tokens:
            subtokens = tokenizer.tokenize(token)
            split_tokens += subtokens
        sents.append(split_tokens)
    return worker_id, sents, batch_offset

def process(cased_file, output_file, bert_model_type='bert-base-cased', total=180378072, chunk_size=1000000, workers=40):
    results = list(range(workers))
    tokenizer = BertTokenizer.from_pretrained(bert_model_type)
    basic_tokenizer = BasicTokenizer(do_lower_case=False)
    fout = open(output_file, 'w')
    offset = 0
    def merge_fn(result):
        worker_id, tokenized, batch_offset = result
        results[worker_id] = tokenized, batch_offset
    for cased_lines in tqdm(get_chunks(cased_file, chunk_size), total=total//chunk_size):
        pool = multiprocessing.Pool()
        size = (len(cased_lines) // workers) if len(cased_lines) % workers == 0 else ( 1 + (len(cased_lines) // workers))
        for i in range(workers):
            start = i * size
            pool.apply_async(tokenize, args = (cased_lines[start:start+size], tokenizer, basic_tokenizer, i, start), callback = merge_fn)
        pool.close()
        pool.join()
        for lines, batch_offset in results:
            for i, line in enumerate(lines):
                fout.write(' '.join(line) + '\n')
        offset += len(cased_lines)


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
