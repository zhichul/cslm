import torch

from cslm.training.trainer import Trainer
from cslm.training.utils import compute_log_probs_with_mask
from transformers.utils import logging


logger = logging.get_logger(__name__)


def mask_logits_by_language(logits, lang_labels, vocab_mask):
    logit_mask = (1 - lang_labels[:, None]) == vocab_mask.expand(logits.size(0), vocab_mask.size(-1))
    logit_mask = logit_mask[:, None, :].expand(logits.size())
    logits = logits.masked_fill(logit_mask, -1e9)
    return logits

class MLETrainer(Trainer):

    def __init__(self, *args, force_langauge=False, ebm=False, l1_range=None, l2_range=None, vocab_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.add_exposure_pattern("encoder_last_layer")

        # setup for forced langauge training
        self.force_language = force_langauge
        if self.force_language:
            self.l1_range = l1_range
            self.l2_range = l2_range
            self.vocab_size = vocab_size
            self.vocab_lang = torch.full((1, self.vocab_size), -1).to(self.args.device)
            self.vocab_lang[0, l1_range] = 0
            self.vocab_lang[0, l2_range] = 1
        # setup a flag for training ebm interventional training
        self.ebm = ebm

    def training_step(self, model, inputs):
        decoder_last_layer = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
        )
        exposed_tensors = dict(self.model.named_exposed_tensors())

        logits = model.lm_head(hidden_states=decoder_last_layer,
                               attention_mask=inputs["decoder_attention_mask"],
                               encoder_hidden_states=exposed_tensors["base_model.encoder_last_layer"],
                               encoder_attention_mask=inputs["attention_mask"])
        self.model.release_exposed_tensors()


        # compute loss
        batch_size = inputs["decoder_attention_mask"].size(0)

        if self.ebm:
            # sample fr
            assert False
        else:
            if self.force_language:
                logits = mask_logits_by_language(logits, inputs["decoder_language_labels"], self.vocab_lang)

        sent_log_prob = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"], inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_prob.sum() / total_tokens
        loss = - token_log_prob / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.item()