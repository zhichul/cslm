from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict
from itertools import product
import random
import orjson

@dataclass
class SyncRule:
    weight: float
    lhs: str
    rhs: List[List[str]]


class SyncGrammar:

    def __init__(self, rules, num_langs=None):
        self.rules = defaultdict(list)
        self.expansions = defaultdict(list)
        self.aligned_nonterminals = defaultdict(list)
        self.weights = defaultdict(list)
        self.num_langs = num_langs
        if self.num_langs is None and len(rules) > 0:
            self.num_langs = len(rules[0].rhs)
        self.add_rules(rules)

    def add_rule(self, rule):
        lhs, rhs, weight = rule.lhs, rule.rhs, rule.weight
        assert len(rhs) == self.num_langs
        self.rules[lhs].append(rule)
        self.weights[lhs].append(weight)
        self.expansions[lhs].append(rhs)
        self.aligned_nonterminals[lhs].append(
            list(set(sum(([sym for sym in rhsi if not self.is_terminal(sym)] for rhsi in rhs), []))))

    def add_rules(self, rules):
        for rule in rules:
            self.add_rule(rule)

    def is_terminal(self, symbol):
        return symbol not in self.expansions if ":" not in symbol else False

    def gen(self, root="ROOT"):
        if self.is_terminal(root):
            # contrary to the gen function of grammar, the synchronous grammar sampler never calls gen on terminal strings
            assert False
        else:
            results = [None] * self.num_langs
            # sample an synchronous expansion E = (E1, E2)
            weights = self.weights[root]
            i = random.choices(range(len(weights)), weights=weights, k=1)[0]

            # pick out the expansion E and aligned nonterminals of the expansion
            expansions = self.expansions[root]
            aligned_nonterminals = self.aligned_nonterminals[root]
            expansion = expansions[i]
            aligned_nonterminal = aligned_nonterminals[i]

            # generate all the nonterminals of the expansion E
            cache = dict()
            for nt in aligned_nonterminal:
                symbol = nt.split(":")[0]
                cache[nt] = self.gen(root=symbol)

            # realize E1 E2 using the expanded aligned nonterminals
            for i in range(self.num_langs):
                seq = []
                for s in expansion[i]:
                    if s in cache:
                        seq.extend(cache[s][i])
                    else:
                        # terminal
                        seq.append(s)
                results[i] = seq
            return tuple(results)

    def all(self, root="ROOT") -> List[Tuple[List[str]]]:
        if self.is_terminal(root):
            # contrary to the gen function of grammar, the synchronous grammar sampler never calls gen on terminal strings
            assert False
        else:
            weights = self.weights[root]
            expansions = self.expansions[root]
            aligned_nonterminals = self.aligned_nonterminals[root]

            #             results = [[] for lang in range(self.num_langs)]

            for expansion, aligned_nonterminal in zip(expansions, aligned_nonterminals):
                # cache derivations
                bigcache = dict()
                for nt in aligned_nonterminal:
                    symbol = nt.split(":")[0]
                    bigcache[nt] = self.all(root=symbol)
                if len(aligned_nonterminal) == 0:
                    yield expansion
                else:
                    for nts in product(*bigcache.values()):
                        yield_result = [None for lang in range(self.num_langs)]
                        # for every combination of expansion of the nonterminals
                        cache = {name: value for name, value in zip(aligned_nonterminal, nts)}
                        # realize every language with it
                        for i in range(self.num_langs):
                            seq = []
                            for s in expansion[i]:
                                if s in cache:
                                    seq.extend(cache[s][i])
                                else:
                                    # terminal
                                    seq.append(s)
                            yield_result[i] = seq
                        yield yield_result

    def __str__(self):
        d = {
            "rules": self.rules,
            "expansions": self.expansions,
            "aligned_nonterminals": self.aligned_nonterminals,
            "weights": self.weights,
        }
        return orjson.dumps(d, option=orjson.OPT_INDENT_2).decode()

    def get_vocab(self, l=None):
        if l is None:
            return sorted(list(set(
                tok for rules in self.rules.values() for rule in rules for seq in rule.rhs for tok in seq if
                self.is_terminal(tok))))
        else:
            return sorted(list(set(
                tok for rules in self.rules.values() for rule in rules for seq in rule.rhs[l:l + 1] for tok in seq if
                self.is_terminal(tok))))

    def save(self, path):
        with open(path, "wb") as f:
            f.write(orjson.dumps([rule for rules in self.rules.values() for rule in rules]))

    def save_vocab(self, prefix):
        for i in range(self.num_langs):
            with open(f"{prefix}.l{i}", "wt") as f:
                for word in self.get_vocab(l=i):
                    print(word, file=f)
        with open(f"{prefix}.joint", "wt") as f:
            for word in self.get_vocab(l=None):
                print(word, file=f)

    @staticmethod
    def from_file(path):
        with open(path, "rb") as f:
            return SyncGrammar(list(SyncRule(**rule_json) for rule_json in orjson.loads(f.read())))

