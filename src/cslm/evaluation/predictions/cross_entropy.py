import orjson
import torch

from cslm.evaluation.prediction import Prediction
from cslm.training.utils import compute_log_probs_with_mask
from cslm.utils import decode_input, decode_output


class CrossEntropyPrediction(Prediction):

    def __init__(self,
                 model=None,
                args=None,
                eval_dataset=None,
                data_collator=None,
                output_file=None,
                cache_file=None,
                filters=None,
                bos_id=None,
                eos_ids=None,
                pad_id=None,
                vocab_size=None,
                l0_tokenizer=None,
                l1_tokenizer=None,
                l2_tokenizer=None,
                force_langauge=False,
                l1_range=None,
                l2_range=None,
                ):
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
        self.l0_tokenizer = l0_tokenizer
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = l2_tokenizer
        self.l1_vocab_size = len(l1_tokenizer.get_vocab())
        self.l2_vocab_size = len(l2_tokenizer.get_vocab())

        # setup for forced language evaluation
        self.force_language = force_langauge
        self.l1_range = l1_range
        self.l2_range = l2_range
        self.vocab_size = vocab_size
        self.vocab_mask = torch.full((1, self.vocab_size), -1).to(self.args.device)
        self.vocab_mask[0, l1_range] = 0
        self.vocab_mask[0, l2_range] = 1

    def _predict_step(self, model, inputs):
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

        if self.force_language:
            logit_mask = (1 - inputs['decoder_language_labels'][:, None]) == self.vocab_mask.expand(batch_size,
                                                                                                    self.vocab_mask.size(
                                                                                                        -1))
            logit_mask = logit_mask[:, None, :].expand(logits.size())
            logits.masked_fill_(logit_mask, -1e9)

        sent_log_prob = compute_log_probs_with_mask(logits, inputs["decoder_input_ids"],
                                                    inputs["decoder_attention_mask"])
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - batch_size
        token_log_prob = sent_log_prob.sum() / total_tokens
        loss = - token_log_prob
        yield {
                    "input_ids": inputs["input_ids"][0].tolist(),
                    "attention_mask": inputs["attention_mask"][0].tolist(),
                    "decoder_input_ids": inputs["decoder_input_ids"][0].tolist(),
                    "decoder_attention_mask": inputs["decoder_attention_mask"][0].tolist(),
                    "encoder_language_label": inputs["encoder_language_labels"][0].tolist(),
                    "decoder_language_label": inputs["decoder_language_labels"][0].tolist(),
                    "input_length": sum(inputs["attention_mask"][0].tolist()),
                    "decoder_input_length": sum(inputs["decoder_attention_mask"][0].tolist()),
                    "output_ids": inputs["decoder_input_ids"][0].tolist(),
                    "output_length": sum(inputs["decoder_attention_mask"][0].tolist()),
                    "weight": 1.0,
                    "log_prob": sent_log_prob.item(),
                    "cross_entropy": loss.item(),
                    "tok_count": total_tokens.item() - 1,
                }

    def _log_step(self, step, predict_result):
        if self.args.decode_format == "data":
            self.output_file.write(orjson.dumps(predict_result) + "\n".encode("utf-8"))
        elif self.args.decode_format == "human":
            # colorful terminal mode
            num_samples_per_example = 1
            if step % num_samples_per_example == 0:
                # do some extra logging
                src = decode_input(predict_result["input_ids"], self.l0_tokenizer)
                tgt = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size)
                print(f"step: {step}", file=self.output_file)
                print(f"src: {src}", file=self.output_file)
                print(f"ref: {tgt}", file=self.output_file)
                print(f"------------------------", file=self.output_file)
            output = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size)
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


