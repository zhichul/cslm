from cslm.evaluation.prediction import Prediction

class Inspection(Prediction):

    def inspect_and_log(self):
        self.predict_and_log()