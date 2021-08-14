import orjson

from cslm.evaluation.evaluation import Evaluation


class LengthMismatch(Evaluation):

    def eval_step(self, predict_result):
        if self.reduction == "macro":
            self.score += predict_result["weight"] * abs(predict_result["output_length"] - predict_result["decoder_input_length"])
            self.count += predict_result["weight"]
        else:
            raise NotImplementedError

    @property
    def summary(self):
        return {
                "length_mismatch": round(self.score / self.count, 4),
                "n_examples": self.count,
        }

    def log(self, summary):
        if self.args.eval_format == "human":
            print(f"{summary['n_examples']:.2f} weighted examples, "
                  f"with {self.reduction} average length mismatch {summary['length_mismatch']:.2f}",
                  file=self.output_file)
        elif self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))

