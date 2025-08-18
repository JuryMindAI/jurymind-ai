import json
from jurymind.core.prompts.base import (
    build_classifier_prompt,
    build_evaluation_prompt,
    build_generator_prompt,
    build_optimizer_prompt,
)
from jurymind.core.models import (
    BatchClassificationResult,
    PromptOptimizationRunResult,
    ClassificationReport,
    ClassificationResult,
    OptimizationRunResult,
    OptimizationStepResult,
    TaskExample
)
from pydantic_ai import Agent


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
        evaluator_model: str = "openai:gpt-5-mini-2025-08-07",
        max_epochs: int = 10,
        num_workers: int = 1,
        search_type: int = "greedy",
    ):
        """Initialize prompt optimization policy

        Args:
            prompt (str): Prompt to optimize in this policy.
            task_description (str): Description of the task we are optimizing the prompt for.
            model (str, optional): LLM to use for optimizing the prompt. Defaults to "gpt-5-mini-2025-08-07".
            max_epochs (int, optional): Max number of epochs to perform optimization on. Defaults to 10.
            num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
            search_type (str, optional): Which search space algorithm to use for finding optimal prompt. Defaults to "greedy".
        """
        self.original_prompt: str = prompt
        self.task_description: str = task_description
        self.num_workers: int = num_workers
        self.max_epochs: int = max_epochs
        self.agent_model: str = model
        self.evaluator_model: str = evaluator_model
        self.search_type: str = search_type  # greedy, beam
        self.step_history: list = []
        self._modified_prompt: str = self.original_prompt
        self.__classification_agent = Agent(
            self.agent_model, output_type=BatchClassificationResult, retries=3
        )
        self.__evaluation_agent = Agent(
            self.evaluator_model, output_type=ClassificationReport, retries=3
        )

        self.__modification_agent = Agent(
            self.agent_model, output_type=OptimizationStepResult, retries=3
        )

    def run(self, task_examples: list[TaskExample] = None):
        """Run the optimization steps for this policy.

        Args:
            task_examples (_type_, optional): Optional list of TaskExample's. Defaults to None and forces model to generate examples solely based on the task_description.
        """        
        # runs the workflow for this policy
        epoch = 0
        # each step holds the current prompt
        current_prompt = self.prompt
        
        while epoch < self.max_epochs:
            
            self._step()

            examples = []
            ground_truth = []
            if task_examples:
                examples = [x["review"] for x in task_examples]
                ground_truth = [x["label"] for x in task_examples]

            batch_prediction_prompt = build_classifier_prompt(
                prompt=current_prompt,
                batch=json.dumps(
                    examples
                ),  # dont give the model both the example and the labels, I think the AI will cheat
                output_schema=BatchClassificationResult.model_json_schema(),
            )

            batch_prediction_result = self.classification_agent.run_sync(
                batch_prediction_prompt
            ).output

            eval_prompt = build_evaluation_prompt(
                current_prompt,
                self.task_description,
                batch_prediction_result,
                ground_truth,
                ClassificationReport.model_json_schema(),
            )

            eval_result = self.__evaluation_agent.run_sync(eval_prompt).output

            self.step_history.append(current_prompt)

            modfication_prompt = build_optimizer_prompt(
                self.step_history,
                current_prompt,
                eval_result.suggested_changes,
            )

            optimization_step_result = self.__modification_agent.run_sync(
                modfication_prompt
            ).output

            current_prompt = optimization_step_result.modified_prompt
            
            epoch += 1
            
        self._modified_prompt = current_prompt
            
    def get_step_history(self):
        return self.step_history
    
    def get_optimized_prompt(self):
        return self._modified_prompt
            

class GreedyOptimizer:
    pass


class BeamSearchOptimizer:
    pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
