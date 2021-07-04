from transformers.utils import logging

logger = logging.get_logger(__name__)


class Evaluation:

    def __init__(self, prediction=None,
                 args=None,
                 output_file=None,
                 reduction="macro",
                 filters=None):
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
        for predict_result in self.prediction.predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")
        return self.summary