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

def constrained_decoding_bin_selector(*args):
    args = set(args)
    logger.info(f"Filtering cells of constrained_decoding {{{', '.join(map(str, args))}}}")
    return lambda prediction_result: prediction_result["cell_id"] in args

def constrained_decoding_length_selector(*args):
    args = set(args)
    logger.info(f"Filtering lengths of constrained_decoding {{{', '.join(map(str, args))}}}")
    return lambda prediction_result: prediction_result["decoder_input_length"] - 2 in args


def language_agnostic_token_matcher():
    # logger.info(f"Filtering underlining in output by containment in language agnostic reference output.")
    def filter_factory(tgt=None):
        tgt = set(t[:-2] for t in tgt[1:-1]) # get rid of language tag
        def filter(token):
            return token[:-2] in tgt
        return filter
    return filter_factory

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
                  l2_tokenizer=None):
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

    def _predict_step(self, model, inputs):
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
                           vocab_size=self.vocab_size)
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