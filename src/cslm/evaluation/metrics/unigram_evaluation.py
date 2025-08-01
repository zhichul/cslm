import orjson
from nltk.translate.bleu_score import sentence_bleu

from cslm.evaluation.evaluation import Evaluation
from cslm.utils import decode_input, decode_output, untag, precision, recall


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
        self.l1_vocab_size = len(l1_tokenizer.get_vocab())
        self.l2_vocab_size = len(l2_tokenizer.get_vocab())

    def eval_step(self, predict_result):
        ref = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size, join=False, color=False)
        output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size, join=False, color=False)
        ref = untag(ref[1:-1])
        output = untag(output[1:-1])
        score = precision(ref, output)
        if self.reduction == "macro":
            self.score += predict_result["weight"] * score
            self.count += predict_result["weight"]
        elif self.reduction == "micro":
            self.score +=  predict_result["weight"] * len(output) * score
            self.count += predict_result["weight"] * len(output)
        else:
            raise NotImplementedError

    @property
    def summary(self):
        return {
                "unigram_precision": self.score / self.count,
                "n_examples": self.count,
        }

    def log(self, summary):
        if self.args.eval_format == "human":
            print(f"{summary['n_examples']:.2f} weighted examples, with {self.reduction} average unigram precision {summary['unigram_precision']:.2f}.", file=self.output_file)
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))

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
        self.l1_vocab_size = len(l1_tokenizer.get_vocab())
        self.l2_vocab_size = len(l2_tokenizer.get_vocab())

    def eval_step(self, predict_result):
        ref = decode_output(predict_result["decoder_input_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size, join=False, color=False)
        output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, self.l1_vocab_size, self.l2_vocab_size, join=False, color=False)
        ref = untag(ref[1:-1])
        output = untag(output[1:-1])
        score = recall(ref, output)
        if self.reduction == "macro":
            self.score += predict_result["weight"] * score
            self.count += predict_result["weight"]
        elif self.reduction == "micro":
            self.score +=  predict_result["weight"] * len(output) * score
            self.count += predict_result["weight"] * len(output)
        else:
            raise NotImplementedError

    @property
    def summary(self):
        return {
                "unigram_recall": self.score / self.count,
                "n_examples": self.count,
        }

    def log(self, summary):
        if self.args.eval_format == "human":
            print(f"{summary['n_examples']:.2f} weighted examples, with {self.reduction} average unigram recall {summary['unigram_recall']:.2f}.", file=self.output_file)
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))