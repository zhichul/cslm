import orjson
import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers.utils import logging

from cslm.evaluation.prediction import Prediction
from cslm.inference.beam_search import beam_search
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

class ConstrainedDecoding(Prediction):

    def __init__(self,
                  model=None,
                  args=None,
                  eval_dataset=None,
                  data_collator=None,
                  output_file=None,
                  cache_file=None,
                  format_filter_factories=None,
                  bos_id=None,
                  eos_ids=None,
                  pad_id=None,
                  vocab_size=None,
                  fn_initial_state=None,
                  fn_update_state=None,
                  fn_assign_bin=None,
                  num_bins=None,
                  do_sample=False,
                  l0_tokenizer=None,
                  l1_tokenizer=None,
                  l2_tokenizer=None):
        super().__init__(model=model,
                         args=args,
                         eval_dataset=eval_dataset,
                         data_collator=data_collator,
                         output_file=output_file,
                         cache_file=cache_file)
        self.bos_id = bos_id
        self.eos_ids = eos_ids
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.fn_initial_state = fn_initial_state
        self.fn_update_state = fn_update_state
        self.fn_assign_bin = fn_assign_bin
        self.num_bins = num_bins
        self.do_sample = do_sample
        self.l0_tokenizer = l0_tokenizer
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = l2_tokenizer

    def _predict_step(self, model, inputs):
        cells = beam_search(model=model,
                           input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           decoder_input_ids=None,
                           decoder_attention_mask=None,
                           max_length=self.args.max_length,
                           num_beams=self.args.decode_num_beams,
                           num_return_sequences=self.args.decode_num_sequences,
                           num_bins=self.num_bins,
                           fn_initial_state=self.fn_initial_state,
                           fn_update_state=self.fn_update_state,
                           fn_assign_bin=self.fn_assign_bin,
                           bos_id=self.bos_id,
                           eos_ids=self.eos_ids,
                           pad_id=self.pad_id,
                           vocab_size=self.vocab_size,
                           do_sample=self.do_sample)
        for b, cell in enumerate(cells):
            log_weights = []
            for score, ids, meta in cell:
                if self.do_sample:
                    log_weights.append(meta["log_weight"])
                else:
                    log_weights.append(0.0)
            weights = torch.softmax(torch.tensor(log_weights), dim=-1).tolist()
            for weight, (score, ids, meta) in zip(reversed(weights), reversed(cell)):
                yield {
                    "input_ids": inputs["input_ids"][0].tolist(),
                    "attention_mask": inputs["attention_mask"][0].tolist(),
                    "decoder_input_ids": inputs["decoder_input_ids"][0].tolist(),
                    "decoder_attention_mask": inputs["decoder_attention_mask"][0].tolist(),
                    "encoder_language_label": inputs["encoder_language_labels"][0].tolist(),
                    "decoder_language_label": inputs["decoder_language_labels"][0].tolist(),
                    "input_length": sum(inputs["attention_mask"][0].tolist()),
                    "decoder_input_length": sum(inputs["decoder_attention_mask"][0].tolist()),
                    "output_ids": ids,
                    "output_length": len(ids),
                    "score": score,
                    "cell_id": b,
                    "weight": weight,
                    "cross_entropy": -meta["log_prob"] / (meta["tok_count"] + 1),
                } | meta

    def _log_step(self, step, predict_result):
        if self.args.decode_format == "data":
            self.output_file.write(orjson.dumps(predict_result) + "\n".encode("utf-8"))
        elif self.args.decode_format == "human":
            # colorful terminal mode
            num_samples_per_example = self.num_bins * self.args.decode_num_sequences
            if step % num_samples_per_example == 0:
                # do some extra logging
                src = decode_input(predict_result["input_ids"], self.l0_tokenizer)
                tgt = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer)
                print(f"step: {step}", file=self.output_file)
                print(f"src: {src}", file=self.output_file)
                print(f"ref: {tgt}", file=self.output_file)
                print(f"------------------------", file=self.output_file)

            # prepare outputs
            output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer)
            ref_toks = untag(decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False,
                                color=False)[1:-1])
            out_toks = untag(decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False,
                                   color=False)[1:-1])
            pcs = precision(ref_toks, out_toks)
            rcl = recall(ref_toks, out_toks)

            # if you want highlight on tokens that match the reference
            # output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, highlight_filter=language_agnostic_token_matcher()(decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False, color=False)))
            line = f"score={predict_result['score']:<7.2f} " \
                   + f"logprob={predict_result['log_prob']:<7.2f} " \
                   + f"ce={-predict_result['log_prob'] / (predict_result['tok_count'] + 1) :<7.2f} " \
                   + background(f"pcs={pcs :<7.2f}", gradient_background(pcs)) + " " \
                   + background(f"rcl={rcl :<7.2f}", gradient_background(rcl)) + " "\
                   + f"l2%={(predict_result['l2_count'] / predict_result['tok_count']) if predict_result['tok_count'] > 0 else 0:<7.2f} " \
                   + (f"cs%={(predict_result['switch_count'] / predict_result['fence_count']) if predict_result['fence_count'] > 0 else 0:<7.2f} " if  "fence_count" in predict_result else "") \
                   + f"weight={predict_result['weight']:<7.2f} " \
                   + f"{output}"
            print(line, file=self.output_file)
            if (step + 1) % self.args.decode_num_sequences == 0:
                print(f"------------------------", file=self.output_file)
            if (step + 1) % num_samples_per_example == 0:
                # print some space
                print(file=self.output_file)