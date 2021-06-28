from cslm.data.generation.pscfg import SyncGrammar, SyncRule


def abstract_grammar_1(nouns=10, verbs=10, adjs=10):
    """
    Template grammar with noun-adj order difference between L1 and L2.
    """
    G = SyncGrammar([
        SyncRule(1.0, "ROOT", [["verb", "VERB:1", "subj", "NP:1", "obj", "NP:2"], ["NP:1", "VERB:1", "NP:2"], ["NP:1", "VERB:1", "NP:2"]]),
        SyncRule(1.0, "ROOT", [["verb", "VERB:1", "obj", "NP:2", "subj", "NP:1"], ["NP:1", "VERB:1", "NP:2"], ["NP:1", "VERB:1", "NP:2"]]),
        SyncRule(1.0, "NP", [["NOUN:1", "mod", "ADJ:1"], ["ADJ:1", "NOUN:1"], ["NOUN:1", "ADJ:1"]])
    ], num_langs=3)
    # noun
    for i in range(nouns):
        G.add_rule(SyncRule(1.0, "NOUN", [[f"noun{i+1}"],[f"noun{i+1}-1"],[f"noun{i+1}-2"]]))
    # verb
    for i in range(verbs):
        G.add_rule(SyncRule(1.0, "VERB", [[f"verb{i+1}"],[f"verb{i+1}-1"],[f"verb{i+1}-2"]]))
    # adjective
    for i in range(adjs):
        G.add_rule(SyncRule(1.0, "ADJ", [[f"adj{i+1}"],[f"adj{i+1}-1"],[f"adj{i+1}-2"]]))
    return G


def abstract_grammar_2(nouns=10, verbs=10, adjs=10):
    """
    Template grammar with noun-adj order difference between L1 and L2, plus having noun phrases.
    """
    G = SyncGrammar([
        SyncRule(1.0, "ROOT", [["verb", "VERB:1", "subj", "NP:1", "obj", "NP:2"], ["NP:1", "VERB:1", "NP:2"],
                               ["NP:1", "VERB:1", "NP:2"]]),
        SyncRule(1.0, "NP", [["NOUN:1", "mod", "ADJ:1"], ["ADJ:1", "NOUN:1"], ["NOUN:1", "ADJ:1"]]),
        SyncRule(1.0, "NP", [["NOUN:1"], ["NOUN:1"], ["NOUN:1"]])
    ], num_langs=3)
    # noun
    for i in range(nouns):
        G.add_rule(SyncRule(1.0, "NOUN", [[f"noun{i + 1}"], [f"noun{i + 1}-1"], [f"noun{i + 1}-2"]]))
    # verb
    for i in range(verbs):
        G.add_rule(SyncRule(1.0, "VERB", [[f"verb{i + 1}"], [f"verb{i + 1}-1"], [f"verb{i + 1}-2"]]))
    # adjective
    for i in range(adjs):
        G.add_rule(SyncRule(1.0, "ADJ", [[f"adj{i + 1}"], [f"adj{i + 1}-1"], [f"adj{i + 1}-2"]]))
    return G