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

    def eval_step(self, predict_result):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def evaluate_and_log(self):
        self.evaluate()
        if self.output_file is not None:
            self.log()

    def evaluate(self):
        for predict_result in self.prediction.predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")