import orjson
import torch

from cslm.evaluation.evaluation import Evaluation
from cslm.inference.beam_search import beam_search
from cslm.utils import decode_output, decode_input


class ConstrainedDecodingEvaluation(Evaluation):

    def __init__(self,
                  model=None,
                  args=None,
                  eval_dataset=None,
                  data_collator=None,
                  output_file=None,
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
        super().__init__(model=model, args=args, eval_dataset=eval_dataset, data_collator=data_collator, output_file=output_file)
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

    def eval_step(self, model, inputs):
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
                    "weight": weight
                } | meta

    def log_step(self, step, eval_result):
        if self.args.decode_format == "data":
            self.output_file.write(orjson.dumps(eval_result) + "\n".encode("utf-8"))
        elif self.args.decode_format == "human":
            # colorful terminal mode
            num_samples_per_example = self.num_bins * self.args.decode_num_sequences
            if step % num_samples_per_example == 0:
                # do some extra logging
                src = decode_input(eval_result["input_ids"], self.l0_tokenizer)
                tgt = decode_output(eval_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer)
                print(f"step: {step}", file=self.output_file)
                print(f"src: {src}", file=self.output_file)
                print(f"ref: {tgt}", file=self.output_file)
                print(f"------------------------", file=self.output_file)
            output = decode_output(eval_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer)
            line = f"score={eval_result['score']:<7.2f} " \
                   + f"logprob={eval_result['log_prob']:<7.2f} " \
                   + f"ce={eval_result['log_prob'] / eval_result['tok_count']:<7.2f} " \
                   + f"l2%={eval_result['l2_count'] / eval_result['tok_count']:<7.2f} " \
                   + f"weight={eval_result['weight']:<7.2f} " \
                   + f"{output}"
            print(line, file=self.output_file)
            if (step + 1) % self.args.decode_num_sequences == 0:
                print(f"------------------------", file=self.output_file)
            if (step + 1) % num_samples_per_example == 0:
                # print some space
                print(file=self.output_file)