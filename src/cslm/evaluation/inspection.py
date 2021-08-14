import code

import torch
from cslm.data.loading.tokenizer_loading import combine_wordlevel_tokenizer
from transformers.utils import logging

from cslm.evaluation.prediction import Prediction
from cslm.utils import integers

logger = logging.get_logger(__name__)


class Inspection(Prediction):

    def __init__(self,
                 model=None,
                 args=None,
                 eval_dataset=None,
                 data_collator=None,
                 output_file=None,
                 cache_file=None,
                 filters=tuple()):
        super().__init__(model=model,
                         args=args,
                         eval_dataset=eval_dataset,
                         data_collator=data_collator,
                         output_file=output_file,
                         cache_file=cache_file,
                         filters=filters)
        self.locals = dict()
        for pattern in self.args.inspect_expose:
            self.model.add_exposure_pattern(pattern)

    def inspect_and_log(self):
        if not self.args.inspect_live:
            self.predict_and_log()
        else:
            self.live_inspect_and_log()

    def live_inspect(self):
        model = self.model
        model.to(self.args.device)
        model.eval()

        # log some stats
        total_eval_batch_size = self.args.per_device_eval_batch_size * self.args.gradient_accumulation_steps

        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = unknown")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_eval_batch_size}")
        logger.info(f"  Total eval batch size (w. parallel, distributed & accumulation) = {total_eval_batch_size}")

        self.output_tokenizer = combine_wordlevel_tokenizer(self.l1_tokenizer, self.l2_tokenizer)
        # eval
        x = ""
        y = ""
        o = ""
        for example_id in integers():
            try:
                while len(x.strip()) == 0 or x.strip().startswith("#"):
                    x = input("Enter X:")
            except: break
            x_tokenization = self.l0_tokenizer.encode(x) # verb verb75 subj noun13 mod adj31 obj noun90
            while True:
                try:
                    while len(y.strip()) == 0 or y.strip().startswith("#"):
                        y = input("Enter Y:")
                except: break
                try:
                    while len(o.strip()) == 0 or o.strip().startswith("#"):
                        o = input("Enter O:")
                except: break
                y_tokenization = self.output_tokenizer.encode(y) # adj31-1 noun13-1 verb75-1 noun90-1
                o_tokenization = self.output_tokenizer.encode(o) # adj31-1 noun13-1 verb75-1 noun90-1
                inputs = dict()
                inputs["input_ids"] = x_tokenization.ids
                inputs["attention_mask"] = x_tokenization.attention_mask
                inputs["decoder_input_ids"] = y_tokenization.ids
                inputs["decoder_attention_mask"] = y_tokenization.attention_mask
                inputs["labels"] = o_tokenization.ids
                inputs["encoder_language_labels"] = -1
                inputs["decoder_language_labels"] = -1
                inputs = self.data_collator([inputs])
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                inputs = {k:v.to(self.args.device) for k,v in inputs.items()}
                for predict_result in self.predict_step(example_id, model, inputs):
                    yield predict_result
                if self.args.inspect_console:
                    self.locals["model"] = self.model
                    self.locals["prev_exposed"] = self.locals.get("exposed", None)
                    self.locals["prev_inputs"] = self.locals.get("inputs", None)
                    self.locals["exposed"] = dict((n,t.detach().cpu()) for n, t in self.model.named_exposed_tensors())
                    self.locals["inputs"] = inputs
                    self.locals["args"] = self.args
                    code.interact(local=self.locals)
                self.model.release_exposed_tensors()
                y = ""
                o = ""
            x = ""
        logger.info("\n\nPrediction completed.\n\n")

    def live_inspect_and_log(self):
        for step, predict_result in enumerate(self.live_inspect()):
            self.log_step(step, predict_result)