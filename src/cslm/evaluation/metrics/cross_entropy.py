import math

import orjson

from cslm.evaluation.evaluation import Evaluation


class CrossEntropyEvaluation(Evaluation):

    def eval_step(self, predict_result):
        if self.reduction == "macro":
            self.score += predict_result["weight"] *predict_result["cross_entropy"]
            self.count += predict_result["weight"]
        elif self.reduction == "micro":
            self.score +=  predict_result["weight"] * (predict_result["tok_count"] + 1) * predict_result["cross_entropy"]
            self.count += predict_result["weight"] * (predict_result["tok_count"] + 1)
        else:
            raise NotImplementedError

    @property
    def summary(self):
        return {
                "cross_entropy": self.score / self.count,
                "n_examples": self.count,
        }

    def log(self, summary):
        if self.args.eval_format == "human":
            print(f"{summary['n_examples']:.2f} weighted examples, "
                  f"with {self.reduction} average entropy (token level) {summary['cross_entropy']:.2f}, "
                  f"perplexity {math.exp(summary['cross_entropy']):.1f}.", file=self.output_file)
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))