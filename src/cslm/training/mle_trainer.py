import torch
from cslm.utils import seq_norm

from cslm.inference.sample import sample
from cslm.training.trainer import Trainer
from cslm.training.utils import compute_log_probs_with_mask, mask_logits_by_language, logit_bias_by_language
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MLETrainer(Trainer):

    def __init__(self, *args, force_langauge=False, l1_range=None, l2_range=None, vocab_size=None, ebm=False, bos_id=None, eos_ids=None, pad_id=None, specialize_decoder_by_language=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.add_exposure_pattern("encoder_last_layer")
        self.model.add_exposure_pattern("attn_weights")

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
        self.bos_id = bos_id
        self.eos_ids = eos_ids
        self.pad_id = pad_id
        # specialize decoder
        self.specialize_decoder_by_language = specialize_decoder_by_language

    def training_step(self, model, inputs):
        if self.ebm:
            return self.training_ebm(model, inputs)
        else:
            return self.training_autoregressive(model, inputs)


    def training_autoregressive(self, model, inputs):
        if self.specialize_decoder_by_language:
            language_ids = inputs["decoder_language_labels"][:,None].expand_as(inputs["input_ids"])
        else:
            language_ids = None
        decoder_last_layer = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            decoder_language_ids=language_ids
        )
        exposed_tensors = dict(self.model.named_exposed_tensors())

        logits = model.lm_head(hidden_states=decoder_last_layer,
                               attention_mask=inputs["decoder_attention_mask"],
                               encoder_hidden_states=exposed_tensors["base_model.encoder_last_layer"],
                               encoder_attention_mask=inputs["attention_mask"])
        self.model.release_exposed_tensors()

        # compute loss
        batch_size = inputs["decoder_attention_mask"].size(0)

        if self.force_language:
            logits = mask_logits_by_language(logits, inputs["decoder_language_labels"], self.vocab_lang)

        sent_log_probs = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"], inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_probs.sum() / total_tokens
        loss = - token_log_prob / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.item()

    def training_ebm(self, model, inputs):
        model.eval() # TODO: this is temporary disables dropout to reduce variance
        if self.specialize_decoder_by_language:
            language_ids = inputs["decoder_language_labels"][:,None].expand_as(inputs["input_ids"])
        else:
            language_ids = None
        decoder_last_layer = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            laguage_ids=language_ids
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
        sent_log_prob = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"], inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_prob.sum() / total_tokens
        term1_loss = - token_log_prob / self.args.gradient_accumulation_steps

        # the second term of the log prob
        # sample from proposal
        if self.force_language:
            logit_bias = logit_bias_by_language(inputs["decoder_language_labels"], vocab_mask=self.vocab_lang)
        else:
            logit_bias = None
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
        attention_mask = attention_mask[:, None, ...].expand(attention_mask.size(0), self.args.monte_carlo_num_sequences,
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
        (term1_loss + term2_loss).backward()
        # for logging
        loss = term1_loss
        # print((term1_loss + term2_loss).item())
        # term1_loss.backward()
        # print((term1_loss).item(), seq_norm(tuple(p.grad for p in model.parameters())).item())
        # term2_loss.backward()
        # print(( term2_loss).item(), seq_norm(tuple(p.grad for p in model.parameters())).item())
        # print("#####")
        return loss.item()
