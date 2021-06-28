from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict
from itertools import product
import random
import orjson


@dataclass
class Rule:
    """
    A weighted grammar rule, where lhs is a single symbol, and rhs is a list of symbols.
    """
    weight: float
    lhs: str
    rhs: List[str]


class Grammar:
    """
    A grammar is a collection of weighted rules. The rules are indexed by their lhs symbols.
    """

    def __init__(self, rules):
        self.rules = defaultdict(list)
        self.expansions = defaultdict(list)
        self.weights = defaultdict(list)
        self.add_rules(rules)

    def add_rule(self, rule):
        lhs, rhs, weight = rule.lhs, rule.rhs, rule.weight
        self.rules[lhs].append(rule)
        self.weights[lhs].append(weight)
        self.expansions[lhs].append(rhs)

    def add_rules(self, rules):
        for rule in rules:
            self.add_rule(rule)

    def is_terminal(self, symbol):
        """
        A symbol is a defined as a terminal if it cannot be expanded.
        """
        return symbol not in self.expansions.keys()

    def gen(self, root="ROOT"):
        """
        Generate a single sequence of symbols.
        """
        if self.is_terminal(root):
            return [root]
        else:
            # sample an expansion
            weights = self.weights[root]
            expansions = self.expansions[root]
            expansion = random.choices(expansions, weights=weights, k=1)[0]

            # recursively expand every symbol of the expansion
            result = []
            for symbol in expansion:
                result.extend(self.gen(root=symbol))
            return result

    def all(self, root="ROOT"):
        if self.is_terminal(root):
            return [[root]]
        else:
            weights = self.weights[root]
            expansions = self.expansions[root]
            results = []
            for expansion in expansions:
                prefixes = [[]]
                for symbol in expansion:
                    extended_prefixes = []
                    # number of prefixes is multiplied by the number of possible expansions for the next symbol
                    for seq in self.all(root=symbol):
                        for prefix in prefixes:
                            extended_prefixes.append(prefix + seq)
                    prefixes = extended_prefixes
                results.extend(prefixes)
            return results



class Integerizer:

    def __init__(self):
        self.i2w = list()
        self.w2i = dict()
        self.add("[UNK]")
        self.add("[BOS]")
        self.add("[EOS]")
        self.add("[PAD]")

    def add(self, w):
        if w not in self.w2i:
            self.i2w.append(w)
            self.w2i[w] = len(self.w2i)
        return self.w2i[w]

    def read_dataset(self, d):
        for sent in d:
            for tok in sent:
                self.add(tok)
        return None

    def tok2ind(self, tokens):
        if isinstance(tokens, str):
            return self.w2i.get(tokens, 0)  # return UNK if the token is not in vocab
        else:
            return [self.tok2ind(tok) for (tok) in tokens]

    def ind2tok(self, inds):
        if isinstance(inds, int):
            return self.i2w[inds]
        else:
            return [self.ind2tok(ind) for (ind) in inds]

    def __len__(self):
        return len(self.i2w)