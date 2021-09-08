import orjson
import torch
from cslm.data.loading.tokenizer_loading import combine_wordlevel_tokenizer

from cslm.training.utils import compute_log_probs_with_mask

from cslm.utils import decode_input, decode_output, gradient_background, background
import numpy as np
from cslm.evaluation.inspection import Inspection

class CrossEntropyInspection(Inspection):

    def __init__(self,
                 model=None,
                args=None,
                eval_dataset=None,
                data_collator=None,
                output_file=None,
                cache_file=None,
                l0_tokenizer=None,
                l1_tokenizer=None,
                l2_tokenizer=None,
        ):
        super().__init__(model=model,
                        args=args,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        output_file=output_file,
                        cache_file=cache_file)
        self.l0_tokenizer = l0_tokenizer
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = l2_tokenizer
        self.l1_vocab_size = len(l1_tokenizer.get_vocab())
        self.l2_vocab_size = len(l2_tokenizer.get_vocab())
        self.combined_tokenizer = combine_wordlevel_tokenizer(self.l1_tokenizer, self.l2_tokenizer)

    def _predict_step(self, model, inputs):
        decoder_last_layer = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
        )
        exposed_tensors = dict(self.model.named_exposed_tensors()) # Intentionally NOT released in this method

        logits = model.lm_head(hidden_states=decoder_last_layer,
                               attention_mask=inputs["decoder_attention_mask"],
                               encoder_hidden_states=exposed_tensors["base_model.encoder_last_layer"],
                               encoder_attention_mask=inputs["attention_mask"])
        sent_log_prob = compute_log_probs_with_mask(logits, inputs["labels"],
                                                    inputs["decoder_attention_mask"])
        token_log_prob = compute_log_probs_with_mask(logits, inputs["labels"],
                                                    inputs["decoder_attention_mask"], reduce=False)
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - 1
        loss = - sent_log_prob.sum() / total_tokens

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
                    "tok_count": sum(inputs["decoder_attention_mask"][0].tolist()) - 2,
                    "log_prob": sent_log_prob.item(),
                    "tok_log_prob": token_log_prob[0].tolist(),
                    "cross_entropy": loss.item(),
                    "labels": inputs["labels"][0].tolist()
                }

    def _log_step(self, step, predict_result):
        if self.args.inspect_format == "data":
            self.output_file.write(orjson.dumps(predict_result) + "\n".encode("utf-8"))
        elif self.args.inspect_format == "human":
            # colorful terminal mode
            num_samples_per_example = 1
            # do some extra logging
            src = decode_input(predict_result["input_ids"], self.l0_tokenizer)
            tgt = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size)
            o = decode_output(predict_result["labels"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size)
            olist = decode_output(predict_result["labels"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size,
                          self.l2_vocab_size,join=False)
            print(f"step: {step}", file=self.output_file)
            print(f"x: {src}", file=self.output_file)
            print(f"y: {tgt}", file=self.output_file)
            print(f"o: {o} log_prob={predict_result['log_prob']:<7.2f} tok_log_prob={','.join([f'{lp:<.2f}' for lp in predict_result['tok_log_prob']])}", file=self.output_file)
            logits_by_lang = predict_result.get('logits_by_lang',None)
            if logits_by_lang is not None:
                logits_display = [[logits_by_lang[i][0][t][id] for t, id in enumerate(predict_result["labels"][1:])] for i in range(2)]
                print(f"lg: {', '.join([f'{olist[t+1]} :' + '/'.join([f'{p:.2f}({max(logits_by_lang[i][0][t],):.2f}:{self.combined_tokenizer.id_to_token(np.argmax(logits_by_lang[i][0][t]))})' for i, p in enumerate(pair)]) for t, pair in enumerate(zip(*logits_display))])}")
            print(f"------------------------\n", file=self.output_file)


class DualActivationCrossEntropy(CrossEntropyInspection):

    def _predict_step(self, model, inputs):
        langs = [0, 1]
        logits_by_lang = [None, None]
        exposed = []
        for lang in langs:
            decoder_last_layer = model.base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=inputs["decoder_input_ids"],
                decoder_attention_mask=inputs["decoder_attention_mask"],
                decoder_language_ids=inputs["decoder_input_ids"].new_full(inputs["decoder_input_ids"].size(), lang)
            )
            exposed_tensors = dict(self.model.named_exposed_tensors())

            logits_by_lang[lang] = model.lm_head(hidden_states=decoder_last_layer,
                                   attention_mask=inputs["decoder_attention_mask"],
                                   encoder_hidden_states=exposed_tensors["base_model.encoder_last_layer"],
                                   encoder_attention_mask=inputs["attention_mask"])
            exposed_tensors = dict(self.model.named_exposed_tensors())
            exposed.append(exposed_tensors)
            model.release_exposed_tensors()
        logits = torch.stack(logits_by_lang, dim=-1)
        logits = torch.logsumexp(logits, dim=-1)
        sent_log_prob = compute_log_probs_with_mask(logits, inputs["labels"],
                                                    inputs["decoder_attention_mask"])
        token_log_prob = compute_log_probs_with_mask(logits, inputs["labels"],
                                                     inputs["decoder_attention_mask"], reduce=False)
        total_tokens = inputs["decoder_attention_mask"].to(dtype=logits.dtype).sum() - 1
        loss = - sent_log_prob.sum() / total_tokens

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
            "tok_count": sum(inputs["decoder_attention_mask"][0].tolist()) - 2,
            "log_prob": sent_log_prob.item(),
            "tok_log_prob": token_log_prob[0].tolist(),
            "cross_entropy": loss.item(),
            "labels": inputs["labels"][0].tolist(),
            "logits_by_lang": [lg.tolist() for lg in logits_by_lang],
            "exposed": exposed
        }