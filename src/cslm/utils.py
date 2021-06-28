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