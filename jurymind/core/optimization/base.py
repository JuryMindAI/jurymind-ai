
class BasePolicy:
    """Base policy class all polcies inherit from"""

    pass


class BaseOptimizer:
    """Base optimizer class all optimizers inherit from"""

    def run(self, input):
        pass


class BasePipeline:
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


class PromptOptimizationPolicy(BasePolicy):
    """
    Policy to optimize a prompt to a specific task.
    """

    def __init__(
        self,
        prompt: str,
        task_description: str,
        model: str = "openai:gpt-5-mini-2025-08-07",
        max_epochs: int = 10,
        num_workers: int = 1,
        search_type: int = "greedy",
    ):
        """Initialize prompt optimization policy

        Args:
            optimization_job_config (PromptOptimizationRequest): _description_
            model (str, optional): LLM to use for optimizing the prompt. Defaults to "gpt-5-mini-2025-08-07".
            max_epochs (int, optional): Max number of epochs to perform optimization on. Defaults to 10.
            num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
            search_type (str, optional): Which search space algorithm to use for finding optimal prompt. Defaults to "greedy".
        """
        self.prompt: str = prompt
        self.task_description: str = task_description
        self.num_workers: int = num_workers
        self.max_epochs: int = max_epochs
        self.model: str = model
        self.search_type: str = search_type  # greedy, beam
        self._step_history: list

    def _step(self):
        """
        Perform a single step update for the policy
        """
        return None

    def run(self, task_examples=None):
        # runs the policy
        i = 0
        while i < self.max_epochs:
            i += 1
            self._step()

            examples = [x["review"] for x in dataset]
            ground_truth = [x["label"] for x in dataset]

            batch_prediction_prompt = __build_classifier_prompt(
                prompt=optimization_request.prompt,
                batch=json.dumps(
                    examples
                ),  # dont give the model both the example and the labels, I think the AI will cheat
                output_schema=BatchClassificationResult.model_json_schema(),
            )


# class PromptOptimizer:
#     def __init__(self, policy: PromptOptimizationPolicy):


class GreedyOptimizer:
    pass


class BeamSearchOptimizer:
    pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
