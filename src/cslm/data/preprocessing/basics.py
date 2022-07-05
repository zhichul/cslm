from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

WS = Whitespace()
NM = Sequence([NFD(), Lowercase(), StripAccents()])

def normalize(s):
    return NM.normalize_str(s)

def tokenize_mixed(s):
    tokens = [pair[0] for pair in WS.pre_tokenize_str(s)]
    output = []
    for token in tokens:
        indices = list(filter(lambda i: token[i].isascii() != token[i-1].isascii() ,list(range(len(token)))))
        indices = [0] + indices + [len(token)]
        for s,e in zip(indices[:-1], indices[1:]):
            subtoken = token[s:e]
            if subtoken.isascii():
                output.extend(tokenize_en(subtoken))
            else:
                output.extend(tokenize_zh(subtoken))
    return output

def tokenize_en(s):
    return [pair[0] for pair in WS.pre_tokenize_str(s)]

def tokenize_zh(s):
    return [c for c in s]