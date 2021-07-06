import orjson
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Evaluation:

    def __init__(self, prediction=None,
                 args=None,
                 output_file=None,
                 reduction="macro",
                 filters=tuple()):
        self.prediction = prediction
        self.args = args
        self.output_file = output_file
        if reduction not in {"micro", "macro", "none"}:
            raise ValueError(f"Reduction must be one of {{'micro', 'macro', 'none'}}, got {reduction}")
        self.reduction = reduction
        self.filters = filters
        self.score = 0
        self.count = 0

    def eval_step(self, predict_result):
        raise NotImplementedError

    @property
    def summary(self):
        raise NotImplementedError

    def log(self, summary):
        raise NotImplementedError

    def evaluate_and_log(self):
        summary = self.evaluate()
        if self.output_file is not None:
            self.log(summary)

    def evaluate(self):
        self.score = 0
        self.count = 0
        for predict_result in self.prediction.lazy_predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")
        return self.summary


class EvaluationList(Evaluation):

    def __init__(self, prediction=None,
                 args=None,
                 output_file=None,
                 filters=tuple()):
        super().__init__(prediction=prediction,
                 args=args,
                 output_file=output_file,
                 filters=filters)
        del self.reduction
        del self.score
        del self.count
        self.evaluations = []

    def add_evaluation(self, name, evaluation):
        self.evaluations.append((name, evaluation))

    def eval_step(self, predict_result):
        for name, ev in self.evaluations:
            ev.eval_step(predict_result)

    @property
    def summary(self):
        summary = dict()
        for name, ev in self.evaluations:
            summary[f"{name}"] = ev.summary
        return summary

    def log(self, summary):
        if self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))
        elif self.args.eval_format == "human":
            for name, ev in self.evaluations:
                self.output_file.write(f"{name}: ")
                ev.log(ev.summary)
        else:
            raise NotImplementedError

    def evaluate(self):
        for name, ev in self.evaluations:
            ev.score = 0
            ev.count = 0
        for predict_result in self.prediction.lazy_predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")
        return self.summary