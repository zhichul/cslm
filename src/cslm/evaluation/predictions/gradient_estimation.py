import math
from functools import reduce

import orjson
import torch

from cslm.evaluation.inspection import Inspection
from cslm.training.mle_trainer import logit_bias_by_language
from cslm.training.utils import compute_log_probs_with_mask
from nltk.translate.bleu_score import sentence_bleu
from transformers.utils import logging

from cslm.evaluation.prediction import Prediction
from cslm.inference.beam_search import beam_search
from cslm.inference.sample import sample
from cslm.utils import decode_output, decode_input, untag, background, gradient_background, recall, precision, seq_add, \
    seq_scale, seq_sub, seq_mul, seq_sum

logger = logging.get_logger(__name__)

class GradientEstimation(Prediction):

    def __init__(self,
                  model=None,
                  args=None,
                  eval_dataset=None,
                  data_collator=None,
                  output_file=None,
                  cache_file=None,
                  filters=None,
                  format_filter_factories=None,
                  bos_id=None,
                  eos_ids=None,
                  pad_id=None,
                  vocab_size=None,
                  num_bins=None,
                  l0_tokenizer=None,
                  l1_tokenizer=None,
                  l2_tokenizer=None,
                  l1_range=None,
                  l2_range=None):
        super().__init__(model=model,
                         args=args,
                         eval_dataset=eval_dataset,
                         data_collator=data_collator,
                         output_file=output_file,
                         cache_file=cache_file,
                         filters=filters)
        self.bos_id = bos_id
        self.eos_ids = eos_ids
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.l0_tokenizer = l0_tokenizer
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = l2_tokenizer
        self.l1_vocab_size = len(l1_tokenizer.get_vocab())
        self.l2_vocab_size = len(l2_tokenizer.get_vocab())
        self.l1_range = l1_range
        self.l2_range = l2_range
        self.vocab_size = vocab_size
        self.vocab_lang = torch.full((1, self.vocab_size), -1).to(self.args.device)
        self.vocab_lang[0, l1_range] = 0
        self.vocab_lang[0, l2_range] = 1

    def _predict_step(self, model, inputs):
        grads = []
        for i in range(self.args.decode_repetitions):
            grads.append(self.gradient_estimator(model, inputs))
        grad_sum = reduce(seq_add,grads,None)
        grad_mean = seq_scale(grad_sum, 1/len(grads))
        grad_diffs = [seq_sub(grad, grad_mean) for grad in grads]
        grad_diffs_square = [seq_mul(diff, diff) for diff in grad_diffs]
        grad_diffs_square_sum = reduce(seq_add, grad_diffs_square, None)
        variances = seq_scale(grad_diffs_square_sum, 1/(len(grads)-1))
        variance = seq_sum(variances) # trace of covariance matrix
        yield {
            "variance": variance.item(),
        }

    def gradient_estimator(self, model, inputs):
        model.eval()
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

        # the first term of the log prob
        sent_log_prob = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"],
                                                    inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_prob.sum() / total_tokens
        term1_loss = - token_log_prob / self.args.gradient_accumulation_steps

        # the second term of the log prob
        # sample from proposal
        logit_bias = logit_bias_by_language(inputs["decoder_language_labels"], vocab_mask=self.vocab_lang)
        proposals = sample(model=model,
                           input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           decoder_input_ids=None,
                           decoder_attention_mask=None,
                           max_length=self.args.max_length,
                           num_return_sequences=self.args.monte_carlo_num_sequences,
                           bos_id=self.bos_id,
                           eos_ids=self.eos_ids,
                           pad_id=self.pad_id,
                           vocab_size=self.vocab_size,
                           logit_bias=logit_bias
                           )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_ids = input_ids[:, None, ...].expand(input_ids.size(0), self.args.monte_carlo_num_sequences,
                                                   *input_ids.size()[1:])  # batch bin beam seq
        attention_mask = attention_mask[:, None, ...].expand(attention_mask.size(0),
                                                             self.args.monte_carlo_num_sequences,
                                                             *attention_mask.size()[1:])  # batch bin beam seq
        decoder_input_ids = proposals["decoder_input_ids"]
        decoder_attention_mask = proposals["decoder_attention_mask"]
        proposal_last_layer = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        exposed_tensors = dict(self.model.named_exposed_tensors())
        logits = model.lm_head(hidden_states=proposal_last_layer,
                               attention_mask=decoder_attention_mask,
                               encoder_hidden_states=exposed_tensors["base_model.encoder_last_layer"],
                               encoder_attention_mask=attention_mask)
        self.model.release_exposed_tensors()
        proposal_log_prob = compute_log_probs_with_mask(logits, decoder_input_ids, decoder_attention_mask)
        log_numerator = proposal_log_prob.tolist()
        log_denominator = proposals["log_probs"]
        log_weights = torch.tensor(log_numerator, device=self.args.device) - torch.tensor(log_denominator,
                                                                                          device=self.args.device)
        weights = torch.softmax(log_weights, dim=-1)
        term2_loss = (weights * proposal_log_prob).sum() / total_tokens / self.args.gradient_accumulation_steps
        loss = term1_loss + term2_loss
        # for logging
        grad = torch.autograd.grad(loss, list(model.parameters()))
        return grad

    def _log_step(self, step, predict_result):
        if self.args.decode_format == "data":
            self.output_file.write(orjson.dumps(predict_result) + "\n".encode("utf-8"))
        elif self.args.decode_format == "human":
            print(predict_result["variance"], file=self.output_file)