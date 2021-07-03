from cslm.training.trainer import Trainer
from cslm.training.utils import compute_log_probs_with_mask
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MLETrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.add_exposure_pattern("encoder_last_layer")

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
        sent_log_prob = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"], inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_prob.sum() / total_tokens
        loss = - token_log_prob / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.item()