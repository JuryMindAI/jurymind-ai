from jurymind.core.models import PromptOptimizationRequest


class BasePolicy:
    """Base policy class all polcies inherit from"""


class BaseOptimizer:
    """Base optimizer class all optimizers inherit from"""

    def __init__(self):
        pass

    def optimize(self, input):
        pass

class BasePipeline:
    def __init__(self, num_iterations, policy):
        self.num_iterations = num_iterations
        self.policy = policy

class OptimizationPipeline(BasePipeline):
    """Higher level container for optimization workflows. The pipeline will run a series of steps and automatically log to mlflow each steps output."""

    def __init__(
        self,
        num_iterations=5,
        steps: list[BasePolicy] = None
    ):
        self.num_iterations: int = num_iterations
        self.steps: list = [] # need to define this
    
    
    def run(self):
        """Run each step in the pipeline and log results to mlflow"""

        
class PromptOptimizationPolicy:
    """
    Policy to optimize a prompt to a specific task.
    """  

    def __init__(
        self,
        optimization_job_config: PromptOptimizationRequest,
        model="openai:gpt-5-mini-2025-08-07",
        max_epochs=10,
        num_workers=1,
        search_type="greedy"
    ):
        """Initialize prompt optimization policy class

        Args:
            optimization_job_config (PromptOptimizationRequest): _description_
            model (str, optional): LLM to use for optimizing the prompt. Defaults to "gpt-5-mini-2025-08-07".
            max_epochs (int, optional): Max number of epochs to perform optimization on. Defaults to 10.
            num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
            search_type (str, optional): Which search space algorithm to use for finding optimal prompt. Defaults to "greedy".
        """               
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.model: str = model
        self.optimization_result = None
        self.search_type: str = search_type  # greedy, beam
        self.task_description: str = optimization_job_config.task_description

    def step(self):
        """_summary_
            Perform a single step for the policy
        """
        
        # set up the agents

class GreedyOptimizer:
    pass

class BeamSearchOptimizer:
    pass

class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
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
        """_summary_

        Args:
            input (_type_): _description_
        """
        pass



class BasePipeline:
    """Base pipeline class all pipelines inherit from"""

    """Base pipeline class all pipelines inherit from"""

    def __init__(self, num_iterations, policy):
        self.num_iterations = num_iterations
        self.policy = policy



class OptimizationPipeline(BasePipeline):
    """Higher level container for optimization workflows. The pipeline will run a series of steps and automatically log to mlflow each steps output."""

    def __init__(self, num_iterations=5, steps: list[BasePolicy] = None):
    def __init__(self, num_iterations=5, steps: list[BasePolicy] = None):
        self.num_iterations: int = num_iterations
        self.steps: list = []  # need to define this

        self.steps: list = []  # need to define this

    def run(self):
        """Run each step in the pipeline and log results to mlflow"""
        pass
        pass


class PromptOptimizationPolicy:
    """
    Policy to optimize a prompt to a specific task.
    """

    def __init__(
        self,
        optimization_job_config: PromptOptimizationRequest,
        model="openai:gpt-5-mini-2025-08-07",
        max_epochs=10,
        num_workers=1,
        search_type="greedy",
    ):
        """Initialize prompt optimization policy class

        Args:
            optimization_job_config (PromptOptimizationRequest): _description_
            model (str, optional): LLM to use for optimizing the prompt. Defaults to "gpt-5-mini-2025-08-07".
            max_epochs (int, optional): Max number of epochs to perform optimization on. Defaults to 10.
            num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
            search_type (str, optional): Which search space algorithm to use for finding optimal prompt. Defaults to "greedy".
        """
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.model: str = model
        self.optimization_result = None
        self.search_type: str = search_type  # greedy, beam
        self.task_description: str = optimization_job_config.task_description

    def step(self):
        """_summary_
        Perform a single step for the policy
        """

        # set up the agents


class GreedyOptimizer:
    pass


class BeamSearchOptimizer:
    pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
