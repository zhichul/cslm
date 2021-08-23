from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import torch
import numpy as np

def sample_indefinitely(data_loader):
    while True:
        for batch in data_loader:
            yield batch

class NumpyWeightedRandomSampler(WeightedRandomSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        total = self.weights.sum()
        self.weights = self.weights / total

    def __iter__(self):
        rand_tensor = np.random.choice(len(self.weights), self.num_samples, p=self.weights, replace=self.replacement)
        return iter(rand_tensor.tolist())


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def compute_log_probs_with_mask(logits, labels, mask, reduce=True):
    # assert (compute_log_probs_with_mask_fast(logits, labels, mask) == compute_log_probs_with_mask_slow(logits, labels, mask)).all()
    return compute_log_probs_with_mask_fast(logits, labels, mask, reduce=reduce)

def compute_probs_with_mask(logits, labels, mask, reduce=True):
    # assert (compute_log_probs_with_mask_fast(logits, labels, mask) == compute_log_probs_with_mask_slow(logits, labels, mask)).all()
    return compute_log_probs_with_mask_fast(logits, labels, mask, reduce=reduce).exp()

def compute_log_probs_with_mask_fast(logits, labels, mask, reduce=True):
    batch_sizes = logits.shape[:-2]
    ctx_size = logits.size(-2)

    shift_logits = logits[..., :-1, :].contiguous()  # (batch_sizes, ctx_size - 1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()  # (batch_sizes, ctx_size - 1, vocab_size)
    shift_mask = mask[..., 1:].contiguous()  # (batch_sizes, ctx_size - 1)
    shift_labels = shift_labels * shift_mask + (-100) * (1 - shift_mask)
    if reduce:
        return - F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").reshape(*batch_sizes, ctx_size-1).sum(dim=-1)
    else:
        return - F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").reshape(*batch_sizes, ctx_size-1)

def compute_log_probs_with_mask_slow(logits, labels, mask):
    batch_size = logits.size(0)
    ctx_size = logits.size(1)
    vocab_size = logits.size(2)

    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, ctx_size, vocab_size)
    labels = labels  # (batch_size, ctx_size)
    mask = mask  # (batch_size, ctx_size)

    shift_log_probs = log_probs[..., :-1, :].contiguous()  # (batch_size, ctx_size - 1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()  # (batch_size, ctx_size - 1, vocab_size)
    shift_mask = mask[..., 1:].contiguous().unsqueeze(-1).to(
        dtype=shift_log_probs.dtype)  # (batch_size, ctx_size - 1, 1)

    masked_log_probs = shift_log_probs * shift_mask  # (batch_size, ctx_size - 1, vocab_size)
    flattened_log_probs = masked_log_probs.reshape(batch_size * (ctx_size - 1),
                                                   vocab_size)  # (batch_size * (ctx_size - 1), vocab_size)
    flattened_labels = shift_labels.reshape(batch_size * (ctx_size - 1))  # (batch_size * (ctx_size - 1))

    data_log_probs = flattened_log_probs[torch.arange(batch_size * (ctx_size - 1)), flattened_labels].reshape(
        batch_size, ctx_size - 1)
    return data_log_probs.sum(dim=-1)