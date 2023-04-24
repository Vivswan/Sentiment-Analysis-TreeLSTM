# tree object from stanfordnlp/treelstm
from typing import List, Optional


class Tree:
    def __init__(self):
        self.parent: Optional[Tree] = None
        self.num_children: int = 0
        self.children: List[Tree] = list()
        self.idx: Optional[int] = None  # node index for SST
        self.gold_label: Optional[int] = None  # node label for SST
        self.output: Optional[int] = None  # output node for SST

        
    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
