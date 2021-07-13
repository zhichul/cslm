from collections import defaultdict

import orjson

from cslm.evaluation.evaluation import Evaluation
from cslm.utils import decode_output, untag, syn_pos


class SyntheticNextWordPOS(Evaluation):

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
        self.reset()

    def reset(self):
        self.score = defaultdict(lambda: defaultdict(float))
        self.count = defaultdict(float)

    def eval_step(self, predict_result):
        output = decode_output(predict_result["output_ids"], self.l1_tokenizer, self.l2_tokenizer, join=False, color=False)[1:]
        output = syn_pos(output)
        for (prev, next) in zip(output, output[1:]):
            if self.reduction == "micro":
                self.score[prev][next] += predict_result["weight"] * 1
                self.count[prev] += predict_result["weight"]
            else:
                raise NotImplementedError

    @property
    def summary(self):
        results = {
            "n_examples": sum(self.count.values())
        }
        for prev, v in self.score.items():
            results[prev] = dict()
            total_count = self.count[prev]
            for next, next_count in v.items():
                results[f"{prev}"][f"{next}"] = next_count / total_count

        return results

    def log(self, summary):
        if self.args.eval_format == "human":
            print(f"{summary['n_examples']:.2f} weighted examples, with the following transition:", file=self.output_file)
            for stat, v in summary.items():
                if stat == "n_examples": continue
                else:
                    for target, prob in sorted(v.items(), key=lambda x: x[1], reverse=True):
                        print(f"\t{stat} -> {target} = {prob}")
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))