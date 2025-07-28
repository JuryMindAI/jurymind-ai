from jurymind.core.models import PromptOptimizationRequest
from enum import Enum, auto


class SearchType(Enum):
    GREEDY = auto()
    BEAM = auto()
    RANDOM = auto()


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

    def __init__(
        self,
        optimization_config: PromptOptimizationRequest,
        model: str = "openai:gpt-4.1-mini",
        iterations: int = 1,
        num_workers: int = 1,
        search_type: SearchType = SearchType.GREEDY,
    ):
        """
        __init__ _summary_
        Initializes the PromptOptimization policy

        _extended_summary_

        Args:
            optimization_config (PromptOptimizationRequest): _description_
            model (_type_, optional): _description_. Defaults to "openai:gpt-4.1-mini".
            iterations (int, optional): _description_. Defaults to 1.
            num_workers (int, optional): _description_. Defaults to 1.
            search_type (SearchType, optional): _description_. Defaults to SearchType.GREEDY.
        """
        self.num_workers: int = num_workers
        self.iterations: int = iterations
        self.model: str = model
        self.optimization_result = None
        self.optimization_request = None
        self.search_type = search_type  # grid, random, beam
        self.task_description: str = optimization_config.task_description

    def optimize(self):
        """_summary_
        Optimization call for the policy given an initial prompt and optional examples
        """
        step = 0

        # iterate till we hit max epochs
        while step < self.max_epochs:
            pass

        return next_prompt_step, step_history


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
