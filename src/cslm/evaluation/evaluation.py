import orjson
from transformers.utils import logging

from cslm.utils import flatten_dict

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
        self.reset()
        for predict_result in self.prediction.lazy_predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")
        return self.summary

    def reset(self):
        self.score = 0
        self.count = 0


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
                ev.mantissa(ev.summary)
        else:
            raise NotImplementedError

    def reset(self):
        for name, ev in self.evaluations:
            ev.reset()

    def evaluate(self):
        self.reset()
        for predict_result in self.prediction.lazy_predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")
        return self.summary

def merge_summaries(*summaries):
    assert all(summary.keys() == summaries[0].keys() for summary in summaries)
    n_examples = sum(map(lambda x: x["n_examples"], summaries))
    return {
        key: (n_examples if key == "n_examples" else (sum(map(lambda x: x["n_examples"] * x[key], summaries)) / n_examples)) for key in summaries[0].keys()
    }

def aggregate_summaries(d):
    if "evaluation_obj" in d:
        return d["evaluation_obj"].summary
    else:
        summaries = []
        for dprime in d.values():
            summaries.append(aggregate_summaries(dprime))
        return merge_summaries(*summaries)

def all_breakdowns(d, parent_breakdown="overall", sep=","):
    results = [(parent_breakdown, aggregate_summaries(d))]
    for feature in d:
        if feature == "evaluation_obj":
            continue
        else:
            results.extend(all_breakdowns(d[feature], parent_breakdown=sep.join([parent_breakdown, feature])))
    return results

class BreakdownEvaluation(Evaluation):

    def __init__(self, prediction=None,
                 args=None,
                 output_file=None,
                 filters=tuple(),
                 factory=None):
        super().__init__(prediction=prediction,
                 args=args,
                 output_file=output_file,
                 filters=filters)
        del self.reduction
        self.evaluations = dict()
        self.breakdowns = list()
        self.factory = factory

    def add_breakdown(self, key):
        self.breakdowns.append(key)

    def eval_step(self, predict_result):
        features = [f"{key}={predict_result[key]}" for key in self.breakdowns]
        d = self.evaluations
        for feature in features:
            if feature not in d:
                d[feature] = dict()
            d = d[feature]
        if "evaluation_obj" not in d:
            eva = self.factory()
            eva.reset()
            d["evaluation_obj"] = eva
        d["evaluation_obj"].eval_step(predict_result)

    @property
    def summary(self):
        return dict(all_breakdowns(self.evaluations))

    def log(self, summary):
        if self.args.eval_format == "data":
            self.output_file.write(orjson.dumps(summary) + "\n".encode("utf-8"))
        elif self.args.eval_format == "human":
            for k, v in summary.items():
                self.output_file.write(f"{k}: {v}" + "\n")
        else:
            raise NotImplementedError

    def reset(self):
        for ev in flatten_dict(self.evaluations).values():
            ev.reset()

    def evaluate(self):
        self.reset()
        for predict_result in self.prediction.lazy_predict_and_log():
            if all(predictate(predict_result) for predictate in self.filters):
                self.eval_step(predict_result)
        logger.info("\n\nEvaluation completed.\n\n")
        return self.summary