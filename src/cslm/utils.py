import re

from scipy.stats import gumbel_r
from transformers.utils import logging
import random
import numpy as np
import torch

logger = logging.get_logger(__name__)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seq_dot(s1, s2):
    if not isinstance(s1, tuple) or not isinstance(s2, tuple) or len(s1) != len(s2):
        raise ValueError(f"{len(s1)}, {len(s2)}")
    cum = 0
    for m1, m2 in zip(s1, s2):
        if m1 is None and m2 is None:
            logger.warning("skipping paired None tensors in seq_dot")
            continue
        assert m1.size() == m2.size()
        cum += (m1 * m2).sum()
    return cum


def seq_norm(s1):
    if not isinstance(s1, tuple):
        raise ValueError()
    cum = 0
    for m1 in s1:
        if m1 is None:
            logger.warning("skipping None tensors in seq_norm")
            continue
        cum += (m1 * m1).sum()
    return cum ** 0.5


def seq_numel(s1):
    if not isinstance(s1, tuple):
        raise ValueError()
    cum = 0
    for m1 in s1:
        if m1 is None:
            logger.warning("skipping None tensors in seq_norm")
            continue
        cum += m1.numel()
    return cum


def seq_add(s1, s2):
    if not isinstance(s1, tuple) or not isinstance(s2, tuple) or len(s1) != len(s2):
        raise ValueError(f"{len(s1)}, {len(s2)}")
    results = tuple()
    for m1, m2 in zip(s1, s2):
        if m1 is None and m2 is None:
            logger.warning("skipping paired None tensors in seq_add")
            continue
        assert m1.size() == m2.size()
        results += (m1 + m2,)
    return results


def seq_sub(s1, s2):
    if not isinstance(s1, tuple) or not isinstance(s2, tuple) or len(s1) != len(s2):
        raise ValueError(f"{len(s1)}, {len(s2)}")
    results = tuple()
    for m1, m2 in zip(s1, s2):
        if m1 is None and m2 is None:
            logger.warning("skipping paired None tensors in seq_sub")
            continue
        assert m1.size() == m2.size()
        results += (m1 - m2,)
    return results


def seq_cosine(s1, s2):
    return seq_dot(s1, s2) / (seq_norm(s1) * seq_norm(s2))


def seq_requires_grad(s1):
    if not isinstance(s1, tuple):
        raise ValueError()
    cum = tuple()
    for m1 in s1:
        if m1 is None:
            logger.warning("skipping None tensors in seq_requires_grad")
            continue
        cum += (m1.requires_grad,)
    return cum


class ImmutableDict(dict):

    def __set__(self, instance, value):
        raise ValueError("Immutable")


def log1mexp(a):
    ret = torch.where(a > -0.693, torch.log(-expm1(a)), log1p(- torch.exp(a)))
    return ret


def log1pexp(a):
    ret = torch.where(a < 18, log1p(torch.exp(a)), a + torch.exp(-a))
    return ret


def log1p(a):
    return torch.log(1 + a)


def expm1(a):
    return torch.exp(a) - 1


def truncated_gumbel(alpha, truncation):
    assert alpha.size(0) == truncation.size(0)
    assert len(alpha.size()) == 2  # batch by items
    assert len(truncation.size()) == 1  # batch

    # By Chris J. Maddison and Danny Tarlow https://cmaddis.github.io/gumbel-machinery
    gumbel = torch.distributions.gumbel.Gumbel(alpha, alpha.new_ones(alpha.size())).sample()
    return -torch.log(torch.exp(-gumbel) + torch.exp(-truncation[:, None]))


def max_gumbel(alpha, max):
    assert alpha.size(0) == max.size(0)
    assert len(alpha.size()) == len(max.size())
    # https://arxiv.org/pdf/1903.06059.pdf appendix b

    gumbel = torch.distributions.gumbel.Gumbel(alpha, alpha.new_ones(alpha.size())).sample()
    Z = gumbel.max(dim=-1, keepdim=True).values

    vi = max - gumbel + log1mexp(gumbel - Z)
    ret = max - torch.clamp(vi, min=0) - log1pexp(-vi.abs())
    return ret


def log_importance_weight(kappa, phi):
    return float(phi - np.log(gumbel_r.sf(kappa, loc=phi, scale=1)))

def red(s):
    return f"\033[31m{s}\033[0m"

def blue(s):
    return f"\033[34m{s}\033[0m"

def pink(s):
    return f"\033[35m{s}\033[0m"

def white(s):
    return f"\033[37m{s}\033[0m"

def bold(s):
    return f"\033[1m{s}\033[0m"

def underline(s):
    return f"\033[4m{s}\033[24m"

def invert(s):
    return f"\033[7m{s}\033[27m"

def highlight(s):
    return f"\033[48;5;10m{s}\033[0m"

def decode_output(ids, l1_tokenizer, l2_tokenizer, join=True, color=True, underline_filter=None, invert_filter=None, highlight_filter=None):
    tokens = []
    bos_token_id = l1_tokenizer.token_to_id("[BOS]")
    eos_token_id = l1_tokenizer.token_to_id("[EOS]")
    l1_vocab_size = len(l1_tokenizer.get_vocab())
    for id in ids:
        ofs = []
        ofe = []
        if id == bos_token_id or id == eos_token_id:
            raw_token = l1_tokenizer.id_to_token(id)
            token = bold(white(raw_token)) if color else raw_token
        elif id >= l1_vocab_size:
            raw_token = l2_tokenizer.id_to_token(id - len(l1_tokenizer.get_vocab()))
            token = red(raw_token) if color else raw_token
        else:
            raw_token = l1_tokenizer.id_to_token(id)
            token = blue(raw_token)  if color else raw_token
        token = underline(token) if underline_filter is not None and underline_filter(raw_token) else token
        token = invert(token) if invert_filter is not None and invert_filter(raw_token) else token
        token = highlight(token) if highlight_filter is not None and highlight_filter(raw_token) else token
        tokens.append(token)
        if id == eos_token_id:
            break
    if join:
        return " ".join(tokens)
    else:
        return tokens

def decode_input(ids, l0_tokenizer, join=True, color=True):
    tokens = []
    bos_token_id = l0_tokenizer.token_to_id("[BOS]")
    eos_token_id = l0_tokenizer.token_to_id("[EOS]")
    for id in ids:
        token = l0_tokenizer.id_to_token(id)
        if id == bos_token_id:
            token = bold(white(token)) if color else token
        else:
            token = pink(token) if color else token
        tokens.append(token)
        if id == eos_token_id:
            break
    if join:
        return " ".join(tokens)
    else:
        return tokens


def untag(tokens):
    return [token[:-2] for token in tokens]

def syn_pos(tokens):
    results = []
    for token in tokens:
        if token in {"[BOS]", "[EOS]", "[PAD]", "[UNK]"}:
            tok = token
        else:
            lang_tag = token[-2:]
            token = token[:-2]
            match = re.match("^([a-zA-Z]+)\d+$", token)
            tok = match.group(1) + lang_tag
        results.append(tok)
    return results

def background(tok, color):
    return f"\033[48;5;{color}m{tok}\033[0m"

def gradient_background(f):
    i = round(f * 10)
    colors = [196, 202, 208, 214, 220, 226, 190, 154, 118, 82, 46]
    color = colors[i]
    return color

def precision(ref, cand):
    if len(cand) == 0:
        return 0
    else:
        ref = set(ref)
        return sum([t in ref for t in cand]) / len(cand)

def recall(ref, cand):
    if len(cand) == 0:
        return 0
    else:
        cand = set(cand)
        return sum([t in cand for t in ref]) / len(ref)