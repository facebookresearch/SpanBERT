import sys
from bitarray import bitarray
from tqdm import tqdm
from collections import defaultdict
nouns = set(['NOUN', 'PROPN', 'PRON'])
verbs = set(['VERB'])
negative_ner_tags = set(['NOTAG', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])
negative_np_tags = set(['NOTAG'])

def bitmap_linenum(raw_file, tagged_file, outfile, negative_tags=negative_np_tags, offset=123391):
    tag_map = []
    tagged = open(tagged_file, errors='replace')
    read_next = True
    line_idx = 0
    #offset  = 0
    with open(raw_file) as f:
        for i, line in tqdm(enumerate(f), total=180254681):
            line = line.strip()
            raw_words = line.split(' ') if line != '' else []
            if read_next:
                tagged_line = tagged.readline()
                #print(tagged_line)
                space_idx = tagged_line.index(' ')
                idx_str, tagged_line = tagged_line[:space_idx], tagged_line[space_idx + 1:]
                tagged_line = tagged_line.strip()
                tagged_words = tagged_line.split(' ') if tagged_line != '' else []
            try:
                line_idx = int(idx_str)
                if i + offset == line_idx:
                    tag_map += [tagged_words[i] not in negative_tags for i in range(1, len(tagged_words), 2) ]
                    read_next = True
                    continue
            except ValueError:
                line_idx = 0
            print('Skipped line {}'.format(i))
            tag_map += [False for i in range(len(raw_words))]
            read_next = (line_idx < (i + 1 + offset))
    print(len(tag_map))
    bitmap = bitarray()
    bitmap.extend(tag_map)
    print('size, prop', bitmap.buffer_info(), bitmap.count() / bitmap.length())
    # with ope
    bitmap.tofile(open(outfile, 'wb'))


def print_distribution(fname):
    counts = defaultdict(int)
    banned = set(['NOTAG', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])
    with open(fname) as f:
        for line in f:
            line = line.strip()
            words = line.split(' ')
            tags = [words[i] for i in range(2, len(words), 2) if words[i] not in banned]
            for tag in tags:
                counts[tag] += 1
    total = sum(counts.values())
    for tag, count in counts.items():
        print(tag, count *100  / total)

if __name__ == '__main__':
    #print_distribution(sys.argv[1])
    bitmap_linenum(sys.argv[1], sys.argv[2], sys.argv[3])

