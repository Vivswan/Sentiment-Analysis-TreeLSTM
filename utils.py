import math
import os
from typing import List, Optional

import torch

from vocab import Vocab


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path + '.txt', encoding="utf-8"))
    with open(path + '.txt', 'r', encoding="utf-8") as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words: List[Optional[str]] = [None] * count
    vectors = torch.zeros(count, dim)
    with open(path + '.txt', 'r', encoding="utf-8") as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            # vectors[idx] = torch.Tensor(map(float, contents[1:]))
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path + '.vocab', 'w', encoding="utf-8") as f:
        for word in words:
            f.write(f'{word}\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile, 'w', encoding="utf-8") as f:
        for token in vocab:
            f.write(token + '\n')


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.Tensor(1, num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor - 1] = 1
    else:
        target[0][floor - 1] = ceil - label
        target[0][ceil - 1] = label - floor
    return target


def count_param(model):
    print('_param count_')
    params = list(model.parameters())
    sum_param = 0
    for p in params:
        sum_param += p.numel()
        print(p.size())
    # emb_sum = params[0].numel()
    # sum_param-= emb_sum
    print('sum', sum_param)
    print('____________')


def print_tree(tree, level):
    indent = ''
    for i in range(level):
        indent += '| '
    line = indent + str(tree.idx)
    print(line)
    for i in range(tree.num_children):
        print_tree(tree.children[i], level + 1)
