from jurymind.core.models import PromptOptimizationRequest


class BasePolicy:
    """Base policy class all polcies inherit from"""


class BaseOptimizer:
    """Base optimizer class all optimizers inherit from"""

    def __init__(self):
        pass

    def optimize(self, input):
        """_summary_

        Args:
            input (_type_): _description_
        """
        pass


class BasePipeline:
    """Base pipeline class all pipelines inherit from"""

    def __init__(self, num_iterations, policy):
        self.num_iterations = num_iterations
        self.policy = policy


class OptimizationPipeline(BasePipeline):
    """Higher level container for optimization workflows. The pipeline will run a series of steps and automatically log to mlflow each steps output."""

    def __init__(self, num_iterations=5, steps: list[BasePolicy] = None):
        self.num_iterations: int = num_iterations
        self.steps: list = []  # need to define this

    def run(self):
        """Run each step in the pipeline and log results to mlflow"""
        pass


class PromptOptimizationPolicy:
    """
    Optimization Policy for tuning prompts to a given task.
    """

    def __init__(
        self,
        optimization_job_config: PromptOptimizationRequest,
        model="",
        max_epochs=1,
        num_workers=1,
    ):
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.model: str = model
        self.optimization_result = None
        self.optimization_request = None
        self.search_type = None  # grid, random, beam
        self.task_description: str = optimization_job_config.task_description

    def step(self):
        """_summary_
        Perform a single step for the policy
        """
        pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
