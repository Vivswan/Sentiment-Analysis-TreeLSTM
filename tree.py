# tree object from stanfordnlp/treelstm
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Tree:
    parent: Optional[Tree] = None
    children: List[Tree] = dataclasses.field(default_factory=list)
    idx: Optional[int] = None  # node index for SST
    gold_label: Optional[int] = None  # node label for SST
    output: Optional[int] = None  # output node for SST

    @property
    def num_children(self):
        return len(self.children)

    @property
    def value(self):
        return self.gold_label

    def state_dict(self):
        return {
            "children": [child.state_dict() for child in self.children],
            "idx": self.idx,
            "gold_label": self.gold_label,
            "output": self.output,
        }

    def load_state_dict(self, state_dict):
        self.children = [Tree().load_state_dict(child) for child in state_dict["children"]]
        for child in self.children:
            child.parent = self

        self.idx = state_dict["idx"]
        self.gold_label = state_dict["gold_label"]
        self.output = state_dict["output"]
        return self

    def add_child(self, child) -> Tree:
        child.parent = self
        self.children.append(child)
        return self

    def size(self) -> int:
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size
        return count

    def depth(self) -> int:
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        return count

    def __repr__(self):
        return f"Tree: value={self.value}, children={self.num_children}"

    def is_leaf(self):
        return self.num_children == 0

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        else:
            return sum([child.get_leaf_nodes() for child in self.children], [])

    def num_leaf_nodes(self):
        return len(self.get_leaf_nodes())

    def get_leaf_values(self, set_idx=False, offset=0):
        result = []
        for idx, leaf in enumerate(self.get_leaf_nodes()):
            if set_idx:
                leaf.idx = idx + offset

            result.append(leaf.gold_label)

        return result
