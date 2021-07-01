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


def decode_output(ids, l1_tokenizer, l2_tokenizer):
    tokens = []
    bos_token_id = l1_tokenizer.token_to_id("[BOS]")
    eos_token_id = l1_tokenizer.token_to_id("[EOS]")
    l1_vocab_size = len(l1_tokenizer.get_vocab())
    for id in ids:
        if id == bos_token_id or id == eos_token_id:
            token = "\033[1;37m" + l1_tokenizer.id_to_token(id) + "\033[0;0m"
        elif id >= l1_vocab_size:
            token = "\033[0;31m" + l2_tokenizer.id_to_token(id - len(l1_tokenizer.get_vocab())) + "\033[0;0m"
        else:
            token = "\033[0;34m" + l1_tokenizer.id_to_token(id) + "\033[0;0m"
        tokens.append(token)
        if id == eos_token_id:
            break
    return " ".join(tokens)


def decode_input(ids, l0_tokenizer):
    tokens = []
    bos_token_id = l0_tokenizer.token_to_id("[BOS]")
    eos_token_id = l0_tokenizer.token_to_id("[EOS]")
    for id in ids:
        if id == bos_token_id:
            token = "\033[1;35m" + l0_tokenizer.id_to_token(id) + "\033[0;0m"
        else:
            token = "\033[0;35m" + l0_tokenizer.id_to_token(id) + "\033[0;0m"
        tokens.append(token)
        if id == eos_token_id:
            break
    return " ".join(tokens)