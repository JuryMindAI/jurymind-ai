# class BaseMetric:
#     pass


# class Accuracy(BaseMetric):
#     def evaluate(self, inputs, outputs, expectations):
#         pass


# class Precision(BaseMetric):
#     def evaluate(self, inputs, output, expectations):
#         pass

import inspect
from functools import wraps


class Evaluator:
    def run(self):
        pass
