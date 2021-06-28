import os

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from cslm.training.trainer import Trainer
from cslm.training.utils import sample_indefinitely, NumpyWeightedRandomSampler, compute_log_probs_with_mask
from transformers.utils import logging
import torch
import tqdm

from cslm.utils import seq_norm

logger = logging.get_logger(__name__)


class MLETrainer(Trainer):

    def training_step(self, model, inputs):
        decoder_last_layer = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
        )
        logits = model.lm_head(decoder_last_layer)

        # compute loss
        batch_size = inputs["decoder_attention_mask"].size(0)
        sent_log_prob = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"], inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_prob.sum() / total_tokens
        loss = - token_log_prob / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.item()