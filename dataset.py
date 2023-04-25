import os
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.utils.data as data
from tqdm import tqdm

import Constants
from tree import Tree


# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences: List[torch.LongTensor] = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences: List[torch.LongTensor] = self.read_sentences(os.path.join(path, 'b.toks'))

        self.ltrees: List[Tree] = self.read_trees(os.path.join(path, 'a.parents'))
        self.rtrees: List[Tree] = self.read_trees(os.path.join(path, 'b.parents'))

        self.labels: torch.Tensor = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size: int = self.labels.size(0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> Tuple[Tree, torch.LongTensor, Tree, torch.LongTensor, torch.Tensor]:
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return ltree, lsent, rtree, rsent, label

    def read_sentences(self, filename) -> List[torch.LongTensor]:
        with open(filename, 'r', encoding="utf-8") as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines(), ascii=True)]
        return sentences

    def read_sentence(self, line) -> torch.LongTensor:
        indices = self.vocab.convert_to_idx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename) -> List[Tree]:
        with open(filename, 'r', encoding="utf-8") as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines(), ascii=True)]
        return trees

    def read_tree(self, line) -> Tree:
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            # if not trees[i-1] and parents[i-1]!=-1:
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    # if trees[parent-1] is not None:
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        
        if root is None:
            raise Exception("No root found")

        return root

    def read_labels(self, filename) -> torch.Tensor:
        with open(filename, 'r', encoding="utf-8") as f:
            labels = map(lambda x: float(x), f.readlines())
            labels = torch.Tensor(labels)
        return labels


# Dataset class for SICK dataset
class SSTDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes, fine_grain, model_name):
        super(SSTDataset, self).__init__()
        self.vocab = vocab
        self.num_classes: int = num_classes
        self.fine_grain: bool = fine_grain
        self.model_name: str = model_name

        temp_sentences = self.read_sentences(os.path.join(path, 'sents.toks'))
        if self.model_name == "dependency":
            temp_trees = self.read_trees(os.path.join(path, 'dparents.txt'), os.path.join(path, 'dlabels.txt'))
        elif self.model_name == "constituency":
            temp_trees = self.read_trees(os.path.join(path, 'parents.txt'), os.path.join(path, 'labels.txt'))
        else:
            raise Exception("Model name not found")

        labels = []
        self.trees = temp_trees
        self.sentences = temp_sentences

        if not self.fine_grain:
            # only get pos or neg
            new_trees = []
            new_sentences = []
            for i in range(len(temp_trees)):
                if temp_trees[i].gold_label != 1:  # 0 neg, 1 neutral, 2 pos
                    new_trees.append(temp_trees[i])
                    new_sentences.append(temp_sentences[i])
            self.trees = new_trees
            self.sentences = new_sentences

        for i in range(0, len(self.trees)):
            labels.append(self.trees[i].gold_label)
        
        self.labels: torch.Tensor = torch.Tensor(labels)  # let labels be tensor
        self.size: int = len(self.trees)

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> Tuple[Tree, torch.Tensor, torch.Tensor]:
        # ltree = deepcopy(self.ltrees[index])
        # rtree = deepcopy(self.rtrees[index])
        # lsent = deepcopy(self.lsentences[index])
        # rsent = deepcopy(self.rsentences[index])
        # label = deepcopy(self.labels[index])
        tree = deepcopy(self.trees[index])
        sent = self.sentences[index]
        label = self.labels[index]
        return tree, sent, label

    def read_sentences(self, filename) -> List[torch.Tensor]:
        with open(filename, 'r', encoding="utf-8") as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines(), ascii=True)]
        return sentences

    def read_sentence(self, line) -> torch.Tensor:
        indices = self.vocab.convert_to_idx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices).type(torch.long)

    def read_trees(self, filename_parents, filename_labels) -> List[Tree]:
        parent = open(filename_parents, 'r', encoding="utf-8").readlines()  # parent node
        label = open(filename_labels, 'r', encoding="utf-8").readlines()  # label node
        trees = [self.read_tree(p_line, l_line) for p_line, l_line in tqdm(zip(parent, label), ascii=True)]
        return trees

    def parse_dlabel_token(self, x):
        if x == '#':
            return None

        if self.fine_grain:  # -2 -1 0 1 2 => 0 1 2 3 4
            return int(x) + 2

        # # -2 -1 0 1 2 => 0 1 2
        tmp = int(x)
        if tmp < 0:
            return 0
        elif tmp == 0:
            return 1
        elif tmp > 0:
            return 2
        else:
            raise ValueError('Cannot parse label token: ' + x)

    def read_tree(self, line, label_line) -> Tree:
        # FIXED: tree.idx, also tree dict() use base 1 as it was in dataset
        # parents is list base 0, keep idx-1
        # labels is list base 0, keep idx-1
        # parents = map(int,line.split()) # split each number and turn to int
        parents = list(map(int, line.split()))  # split each number and turn to int
        trees = dict()  # this is dict
        root = None
        # labels = map(self.parse_dlabel_token, label_line.split())
        labels = list(map(self.parse_dlabel_token, label_line.split()))
        for i in range(1, len(parents) + 1):
            if i in trees.keys() or parents[i - 1] == -1:
                continue

            # for i in range(1,len(list(parents))+1):
            # if not trees[i-1] and parents[i-1]!=-1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break

                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)

                trees[idx] = tree
                tree.idx = idx  # -1 remove -1 here to prevent embs[tree.idx -1] = -1 while tree.idx = 0
                tree.gold_label = labels[idx - 1]  # add node label
                # if trees[parent-1] is not None:

                if parent in trees.keys():
                    trees[parent].add_child(tree)
                    break
                if parent == 0:
                    root = tree
                    break

                prev = tree
                idx = parent

        if root is None:
            raise ValueError('No root node found')

        return root

    @staticmethod
    def read_labels(filename):
        # Not in used
        with open(filename, 'r', encoding="utf-8") as f:
            labels = map(lambda x: float(x), f.readlines())
            labels = torch.Tensor(labels)
        return labels
