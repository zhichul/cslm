import orjson
from nltk.translate.bleu_score import sentence_bleu

from cslm.evaluation.evaluation import Evaluation
from cslm.utils import decode_input, decode_output, untag


class UnigramLanguageAgnosticPrecision(Evaluation):

    def __init__(self,
                 prediction=None,
                 args=None,
                 output_file=None,
                 reduction="macro",
                 filters=None,
                 l0_tokenizer=None,
                 l1_tokenizer=None,
                 l2_tokenizer=None):
        super().__init__(prediction=prediction,
                         args=args,
                         output_file=output_file,
                         reduction=reduction,
                         filters=filters)
        self.l0_tokenizer = l0_tokenizer
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = l2_tokenizer

    def eval_step(self, predict_result):
        ref = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False, color=False)
        output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False, color=False)
        ref = untag(ref[1:-1])
        output = untag(output[1:-1])
        score = sentence_bleu([ref], output, weights=(1, 0, 0, 0))
        if self.reduction == "macro":
            self.score += predict_result["weight"] * score
            self.count += predict_result["weight"]
        elif self.reduction == "micro":
            self.score +=  predict_result["weight"] * len(output) * score
            self.count += predict_result["weight"] * len(output)
        else:
            raise NotImplementedError

    def log(self):
        result = {
                "unigram_precision": self.score / self.count,
                "n_examples": self.count,
            }
        if self.args.eval_format == "human":
            print(f"{result['n_examples']:.2f} weighted examples, with {self.reduction} average unigram precision {result['unigram_precision']:.2f}.", file=self.output_file)
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(result) + "\n".encode("utf-8"))

class UnigramLanguageAgnosticRecall(Evaluation):

    def __init__(self,
                 prediction=None,
                 args=None,
                 output_file=None,
                 reduction="macro",
                 filters=None,
                 l0_tokenizer=None,
                 l1_tokenizer=None,
                 l2_tokenizer=None):
        super().__init__(prediction=prediction,
                         args=args,
                         output_file=output_file,
                         reduction=reduction,
                         filters=filters)
        self.l0_tokenizer = l0_tokenizer
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = l2_tokenizer

    def eval_step(self, predict_result):
        ref = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False, color=False)
        output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False, color=False)
        ref = untag(ref[1:-1])
        output = untag(output[1:-1])
        score = sentence_bleu([output], ref, weights=(1, 0, 0, 0))
        if self.reduction == "macro":
            self.score += predict_result["weight"] * score
            self.count += predict_result["weight"]
        elif self.reduction == "micro":
            self.score +=  predict_result["weight"] * len(output) * score
            self.count += predict_result["weight"] * len(output)
        else:
            raise NotImplementedError

    def log(self):
        result = {
                "unigram_recall": self.score / self.count,
                "n_examples": self.count,
            }
        if self.args.eval_format == "human":
            print(f"{result['n_examples']:.2f} weighted examples, with {self.reduction} average unigram recall {result['unigram_recall']:.2f}.", file=self.output_file)
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(result) + "\n".encode("utf-8"))