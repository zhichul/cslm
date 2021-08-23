import math

import orjson
import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers.utils import logging

from cslm.evaluation.prediction import Prediction
from cslm.inference.beam_search import beam_search
from cslm.inference.sample import sample
from cslm.utils import decode_output, decode_input, untag, background, gradient_background, recall, precision

logger = logging.get_logger(__name__)

class Sampling(Prediction):

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
                  force_langauge=False,
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
        self.force_language = force_langauge
        if self.force_language:
            self.l1_range = l1_range
            self.l2_range = l2_range
            self.vocab_size = vocab_size
            self.l1_bias = torch.zeros((self.vocab_size,), device=self.args.device)
            self.l2_bias = torch.zeros((self.vocab_size,), device=self.args.device)
            self.l1_bias[l2_range] = -float("inf")
            self.l2_bias[l1_range] = -float("inf")

    def _predict_step(self, model, inputs):
        batch_size = inputs["input_ids"].size(0)
        if self.force_language:
            l1_bias = self.l1_bias[None, :].expand(batch_size, self.vocab_size)
            l2_bias = self.l2_bias[None, :].expand(batch_size, self.vocab_size)
            lang_labels = inputs["decoder_language_labels"][:, None].expand(batch_size, self.vocab_size)
            logit_bias = torch.where(lang_labels == 0, l1_bias , l2_bias)
        else:
            logit_bias = None
        outputs = sample(model=model,
                           input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           decoder_input_ids=None,
                           decoder_attention_mask=None,
                           max_length=self.args.max_length,
                           num_return_sequences=self.args.decode_num_sequences,
                           bos_id=self.bos_id,
                           eos_ids=self.eos_ids,
                           pad_id=self.pad_id,
                           vocab_size=self.vocab_size,
                           logit_bias=logit_bias)
        for _, (log_prob, output, attention_mask) in enumerate(zip(outputs["log_probs"][0], outputs["decoder_input_ids"][0], outputs["decoder_attention_mask"][0])):
            yield {
                "input_ids": inputs["input_ids"][0].tolist(),
                "attention_mask": inputs["attention_mask"][0].tolist(),
                "decoder_input_ids": inputs["decoder_input_ids"][0].tolist(),
                "decoder_attention_mask": inputs["decoder_attention_mask"][0].tolist(),
                "encoder_language_label": inputs["encoder_language_labels"][0].tolist(),
                "decoder_language_label": inputs["decoder_language_labels"][0].tolist(),
                "input_length": sum(inputs["attention_mask"][0].tolist()),
                "decoder_input_length": sum(inputs["decoder_attention_mask"][0].tolist()),
                "output_ids": output.tolist(),
                "output_length": sum(attention_mask.tolist()) - 2,
                "score": log_prob,
                "cell_id": 0,
                "weight": math.exp(log_prob),
                "cross_entropy": log_prob / (sum(attention_mask.tolist()) - 1),
                "log_prob":log_prob,
                "tok_count": sum(attention_mask.tolist()) - 2,
            }

    def _log_step(self, step, predict_result):
        if self.args.decode_format == "data":
            self.output_file.write(orjson.dumps(predict_result) + "\n".encode("utf-8"))
        elif self.args.decode_format == "human":
            # colorful terminal mode
            num_samples_per_example = self.args.decode_num_sequences
            if step % num_samples_per_example == 0:
                # do some extra logging
                src = decode_input(predict_result["input_ids"], self.l0_tokenizer)
                tgt = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer,
                                    self.l1_vocab_size, self.l2_vocab_size)
                print(f"step: {step}", file=self.output_file)
                print(f"src: {src}", file=self.output_file)
                print(f"ref: {tgt}", file=self.output_file)
                print(f"------------------------", file=self.output_file)
            output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer,
                                   self.l1_vocab_size, self.l2_vocab_size)
            line = f"logprob={predict_result['log_prob']:<7.3f} " \
                   + f"ce={-predict_result['log_prob'] / (predict_result['tok_count'] + 1):<7.3f} " \
                   + f"weight={predict_result['weight']:<7.3f} " \
                   + f"{output}"
            print(line, file=self.output_file)
            if (step + 1) % self.args.decode_num_sequences == 0:
                print(f"------------------------", file=self.output_file)
            if (step + 1) % num_samples_per_example == 0:
                # print some space
                print(file=self.output_file)