from jurymind.core.models import PromptOptimizationRequest


class BasePolicy:
    """Base class for policy objects"""


class BaseOptimizer:
    """Base optimizer class"""

    def __init__(self):
        pass

    def optimize(self):
        pass

class BasePipeline:
    def __init__(self, num_iterations, policy):
        self.num_iterations = num_iterations
        self.policy = policy

class OptimizationPipeline(BasePipeline):
    """Higher level container for optimization workflows."""

    def __init__(
        self,
        num_iterations=5,
        policy: BasePolicy = None
    ):
        self.num_iterations: int = num_iterations
        self.steps: list = None # need to define this
    def run(self):
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

    def optimize(self):
        """_summary_
        Begins optimization of the requested job.
        """
        pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
