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

def abstract_grammar_3(nouns_per_lang=10, shared_nouns=10, verbs=10, adjs=10):
    """
    Template grammar with noun-adj order difference between L1 and L2, plus having noun phrases.
    """
    G = SyncGrammar([
        SyncRule(1.0, "ROOT", [["verb", "VERB:1", "subj", "NP-1:1", "obj", "NP-1:2"], ["NP-1:1", "VERB:1", "NP-1:2"],
                               ["NP-1:1", "VERB:1", "NP-1:2"]]),
        SyncRule(1.0, "ROOT", [["verb", "VERB:1", "subj", "NP-2:1", "obj", "NP-2:2"], ["NP-2:1", "VERB:1", "NP-2:2"],
                               ["NP-2:1", "VERB:1", "NP-2:2"]]),
        SyncRule(1.0, "NP-1", [["NOUN-1:1", "mod", "ADJ:1"], ["ADJ:1", "NOUN-1:1"], ["NOUN-1:1", "ADJ:1"]]),
        SyncRule(1.0, "NP-2", [["NOUN-2:1", "mod", "ADJ:1"], ["ADJ:1", "NOUN-2:1"], ["NOUN-2:1", "ADJ:1"]]),
        SyncRule(1.0, "NP-1", [["NOUN-1:1"], ["NOUN-1:1"], ["NOUN-1:1"]]),
        SyncRule(1.0, "NP-2", [["NOUN-2:1"], ["NOUN-2:1"], ["NOUN-2:1"]]),
    ], num_langs=3)
    # noun
    for l in range(1,3,1):
        for i in range(nouns_per_lang):
            G.add_rule(SyncRule(1.0, f"NOUN-{l}", [[f"noun{l*1000 + i + 1}"], [f"noun{l*1000 + i + 1}-1"], [f"noun{l*1000 + i + 1}-2"]]))
    for i in range(shared_nouns):
        for l in range(1, 3, 1):
            G.add_rule(SyncRule(1.0, f"NOUN-{l}", [[f"noun{i + 1}"], [f"noun{i + 1}-1"],
                                                   [f"noun{i + 1}-2"]]))
    # verb
    for i in range(verbs):
        G.add_rule(SyncRule(1.0, "VERB", [[f"verb{i + 1}"], [f"verb{i + 1}-1"], [f"verb{i + 1}-2"]]))
    # adjective
    for i in range(adjs):
        G.add_rule(SyncRule(1.0, "ADJ", [[f"adj{i + 1}"], [f"adj{i + 1}-1"], [f"adj{i + 1}-2"]]))
    return G

def abstract_grammar_2t(nouns=10, verbs=10, adjs=10):
    """
    Template grammar with noun-adj order difference between L1 and L2, plus having noun phrases.
    """
    G = SyncGrammar([
        SyncRule(1.0, "ROOT", [["SNP:1", "VP:1"], ["SNP:1", "VP:1"],["SNP:1", "VP:1"]]),
        SyncRule(1.0, "SNP", [["subj", "NP:1"], ["NP:1"], ["NP:1"]]),
        SyncRule(1.0, "ONP", [["obj", "NP:1"], ["NP:1"], ["NP:1"]]),
        SyncRule(1.0, "VP", [["verb", "VERB:1", "ONP:1"], ["VP-1", "VERB:1", "ONP:1"], ["VP-2","VERB:1", "ONP:1"]]),
        SyncRule(1.0, "NP", [["NOUN:1", "mod", "ADJ:1"], ["NP-1", "ADJ:1", "NOUN:1"], ["NP-2", "NOUN:1", "ADJ:1"]]),
        SyncRule(1.0, "NP", [["NOUN:1"], ["NP-1", "NOUN:1"], ["NP-2", "NOUN:1"]])
    ], num_langs=3)
    # noun
    for i in range(nouns):
        G.add_rule(SyncRule(1.0, "NOUN", [[f"noun{i + 1}"], ["N-1", f"noun{i + 1}-1"], ["N-2", f"noun{i + 1}-2"]]))
    # verb
    for i in range(verbs):
        G.add_rule(SyncRule(1.0, "VERB", [[f"verb{i + 1}"], ["V-1", f"verb{i + 1}-1"], ["V-2", f"verb{i + 1}-2"]]))
    # adjective
    for i in range(adjs):
        G.add_rule(SyncRule(1.0, "ADJ", [[f"adj{i + 1}"], ["A-1", f"adj{i + 1}-1"], ["A-2", f"adj{i + 1}-2"]]))
    return G

def abstract_grammar_2t_aligned(nouns=10, verbs=10, adjs=10):
    """
    Template grammar with noun-adj order difference between L1 and L2, plus having noun phrases.
    """
    G = SyncGrammar([
        SyncRule(1.0, "ROOT", [["SNP:1", "VP:1"], ["SNP:1", "VP:1"],["SNP:1", "VP:1"]]),
        SyncRule(1.0, "SNP", [["subj", "NP:1"], ["NP:1"], ["NP:1"]]),
        SyncRule(1.0, "ONP", [["obj", "NP:1"], ["NP:1"], ["NP:1"]]),
        SyncRule(1.0, "VP", [["verb", "VERB:1", "ONP:1"], ["(vp", "VERB:1", "ONP:1", ")vp"], ["(vp","VERB:1", "ONP:1", ")vp"]]),
        SyncRule(1.0, "NP", [["NOUN:1", "mod", "ADJ:1"], ["(np", "ADJ:1", "NOUN:1", ")np"], ["(np", "NOUN:1", "ADJ:1", ")np"]]),
        SyncRule(1.0, "NP", [["NOUN:1"], ["(np", "NOUN:1", ")np"], ["(np", "NOUN:1", ")np"]])
    ], num_langs=3)
    # noun
    for i in range(nouns):
        G.add_rule(SyncRule(1.0, "NOUN", [[f"noun{i + 1}"], ["(n", f"noun{i + 1}-1", ")n"], ["(n", f"noun{i + 1}-2", ")n"]]))
    # verb
    for i in range(verbs):
        G.add_rule(SyncRule(1.0, "VERB", [[f"verb{i + 1}"], ["(v", f"verb{i + 1}-1", ")v"], ["(v", f"verb{i + 1}-2", ")v"]]))
    # adjective
    for i in range(adjs):
        G.add_rule(SyncRule(1.0, "ADJ", [[f"adj{i + 1}"], ["(a", f"adj{i + 1}-1", ")a"], ["(a", f"adj{i + 1}-2", ")a"]]))
    return G

def abstract_grammar_4t_aligned(nouns=10, verbs=10, adjs=10, pronouns=10):
    """
    Template grammar with noun-adj order difference between L1 and L2, plus having noun phrases.
    """
    G = SyncGrammar([
        SyncRule(1.0, "ROOT", [["SNP:1", "VP:1"], ["SNP:1", "VP:1"],["SNP:1", "VP:1"]]),
        SyncRule(1.0, "SNP", [["subj", "NP:1"], ["NP:1"], ["NP:1"]]),
        SyncRule(1.0, "ONP", [["obj", "NP:1"], ["NP:1"], ["NP:1"]]),
        SyncRule(1.0, "VP", [["verb", "VERB:1", "ONP:1"], ["(vp", "VERB:1", "ONP:1", ")vp"], ["(vp","VERB:1", "ONP:1", ")vp"]]),
        SyncRule(1.0, "NP", [["DET:1", "NOM:1"], ["(np", "DET:1", "NOM:1", ")np"], ["(np", "DET:1", "NOM:1", ")np"]]),
        SyncRule(1.0, "NP", [["NNP:1"], ["(np", "NNP:1", ")np"], ["(np", "NNP:1", ")np"]]),
        SyncRule(1.0, "NOM", [["NOUN:1", "mod", "ADJ:1"], ["(nom", "ADJ:1", "NOUN:1", ")nom"], ["(nom", "NOUN:1", "ADJ:1", ")nom"]]),
        SyncRule(1.0, "NOM", [["NOUN:1"], ["(nom", "NOUN:1", ")nom"], ["(nom", "NOUN:1", ")nom"]]),
        SyncRule(1.0, "DET", [["def"],["(det", "the-1", ")det"],["(det", "the-2", ")det"]]),
        SyncRule(1.0, "DET", [["indef"], ["(det", "a-1", ")det"], ["(det", "a-2", ")det"]]),
    ], num_langs=3)
    # noun
    for i in range(nouns):
        G.add_rule(SyncRule(1.0, "NOUN", [[f"noun{i + 1}"], ["(n", f"noun{i + 1}-1", ")n"], ["(n", f"noun{i + 1}-2", ")n"]]))
    # verb
    for i in range(verbs):
        G.add_rule(SyncRule(1.0, "VERB", [[f"verb{i + 1}"], ["(v", f"verb{i + 1}-1", ")v"], ["(v", f"verb{i + 1}-2", ")v"]]))
    # adjective
    for i in range(adjs):
        G.add_rule(SyncRule(1.0, "ADJ", [[f"adj{i + 1}"], ["(a", f"adj{i + 1}-1", ")a"], ["(a", f"adj{i + 1}-2", ")a"]]))
    # adjective
    for i in range(pronouns):
        G.add_rule(
            SyncRule(1.0, "NNP", [[f"name{i + 1}"], ["(nnp", f"name{i + 1}-1", ")nnp"], ["(nnp", f"name{i + 1}-2",  ")nnp"]]))
    return G