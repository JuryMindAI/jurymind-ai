"""
Classes and functions to run different opmtimization tasks.
"""

import mlflow
import random
from mlflow.entities import SpanType

import json

from typing import Callable

from pydantic_ai import Agent
from loguru import logger
from jurymind.core.prompts.base import (
    build_classifier_prompt,
    build_evaluation_prompt,
    build_optimizer_prompt,
)
from jurymind.core.models import (
    BatchClassificationResult,
    ClassificationReport,
    OptimizationStepResult,
    TaskExample,
)


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


class PromptOptimizer(BasePolicy):
    """
    Optimize a prompt to a specific task.
    """

    def __init__(
        self,
        prompt: str,
        task_description: str,
        model: str = "openai:gpt-4.1-mini",
        evaluator_model: str = "openai:gpt-4.1",  # Defaults to more advanced model for evaluations
        max_epochs: int = 5,
        num_workers: int = 1,
        search_type: str = "beam",
        tracking_mlflow: bool = False,
        training_examples: list[TaskExample] = None,
        evaluation_examples: list[TaskExample] = None,
        evaluators: list[Callable] = None,
    ):
        """
        Initialize prompt optimization

        Args:
            prompt (str): Prompt to optimize in this policy.
            task_description (str): Description of the task we are optimizing the prompt for.
            model (str, optional): LLM to use for optimizing the prompt. Defaults to "gpt-5-mini-2025-08-07".
            max_epochs (int, optional): Max number of epochs to perform optimization on. Defaults to 10.
            num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
            search_type (str, optional): Which search space algorithm to use for finding optimal prompt. Defaults to "greedy".
            track_mlflow (bool, optional): Use mlflow tracking. Defaults to False.
            task_examples (list[TaskExample], optional): Optional list of TaskExample's to help generate new examples from. Defaults to None.
            evaluation_examples (list[TaskExample], optional): Optional list of TaskExample's to use as a test set for evaluate the prompts on. Defaults to None
        """
        self.original_prompt: str = prompt
        self._modified_prompt: str | None = None
        self.task_description: str = task_description
        self.num_workers: int = num_workers
        self.max_epochs: int = max_epochs
        self.agent_model_id: str = model
        self.evaluator_model_id: str = evaluator_model
        self.search_type: str = search_type  # greedy, beam
        self._policy_optimization_history: list = []
        self.training_examples: list[TaskExample] = training_examples
        self.evaluation_examples: list[TaskExample] = evaluation_examples
        self.evaluation_functions: list[Callable] = evaluators

        # Setup the agents to be used in this policy workflow
        self.__classification_agent = Agent(
            self.agent_model_id, output_type=BatchClassificationResult, retries=3
        )
        self.__evaluation_agent = Agent(
            self.evaluator_model_id, output_type=ClassificationReport, retries=3
        )

        # self.__generation_agent = Agent(self.agent_model, output_type=)

        self.__modification_agent = Agent(
            self.agent_model_id, output_type=OptimizationStepResult, retries=3
        )

        # self.__tracking_mlflow = tracking_mlflow

    def _candidate_generation(self, prompt, task_description, suggestions=None, n=5):
        """
        Generates a list of candidates to test against

        Args:
            prompt (_type_): _description_
            task_description (_type_): _description_
            examples (_type_, optional): _description_. Defaults to None.
        """
        raise NotImplementedError()

    def _select(self, prompts):
        """
        helper function to select the best candidates

        Args:
            prompts (_type_): _description_
        """

        raise NotImplementedError()

    def __run_evaluations(
        self, model_predictions: list[str], data_expectations: list[str]
    ):
        """
        Runs the evaluation functions, if provided, over the evaluation examples to
        align the prompt changes to the target function.
        """
        results = []
        for func in self.evaluation_functions:
            results.append(func(model_predictions, data_expectations))
        return results

    def __search_space(
        self, space: list, examples: list, ground_truth=None, n=5
    ) -> list:
        """
        Perform beam search

        Args:
            space (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        candidates = []
        for prompt in space:
            # Could maybe multi thread this since each prompt would be

            minibatch_sample = random.sample(examples, n)
            batch_prediction_prompt = build_classifier_prompt(
                prompt=prompt,
                batch=json.dumps(
                    minibatch_sample
                ),  # dont give the model both the example and the labels, the llm may try to cheat.
            )

            batch_prediction_result = self.__classification_agent.run_sync(
                batch_prediction_prompt
            ).output

            # list of evaluation results we need to merge with all the candidates
            evaluation_metric_results = self.__run_evaluations(
                batch_prediction_result.predictions, ground_truth
            )

            eval_report_prompt = build_evaluation_prompt(
                prompt,
                self.task_description,
                evaluation_metric_results,
                batch_prediction_result,
                ground_truth,
                ClassificationReport.model_json_schema(),
            )
            #     # attempt to use LLM to evaluate the ouput results
            eval_report = self.__evaluation_agent.run_sync(eval_report_prompt).output

            modfication_prompt = build_optimizer_prompt(
                self._policy_optimization_history,
                prompt,
                eval_report.suggested_changes,
            )

            optimization_step_result = self.__modification_agent.run_sync(
                modfication_prompt
            ).output

            # TODO: change this to maybe not return all the eval metric results for the given prompt but a mean?
            candidates.append(
                (optimization_step_result.modified_prompt, evaluation_metric_results)
            )

        return candidates

    def run(self):
        """Run the optimization steps for this policy."""
        logger.info("Beginning start of optimization policy execution.")
        # runs the workflow for this policy
        epoch = 1
        # each step holds the current prompt
        # current_prompt = self.original_prompt

        __batch_prompt = [self.original_prompt]

        examples = [x.example for x in self.evaluation_examples]
        ground_truth = [x.label for x in self.evaluation_examples]

        # Loop for n epochs, collecting up the results per pass
        while epoch <= self.max_epochs:
            # what do I do about the eval_results...
            candidates, eval_results = self.__search_space(
                __batch_prompt, examples, ground_truth
            )
            __batch_prompt.extend(candidates)
        #     logger.info("Beginning batch prediction step.")
        #     batch_prediction_result = self.__classification_agent.run_sync(
        #         batch_prediction_prompt
        #     ).output

        #     logger.info("Begining evaluation of batch predictions.")

        #     # eval_result = self.__evaluation_agent.run_sync(eval_prompt).output
        #     # run evaluators over the evaluation dataset if its provided
        #     # eval_results = None
        #     # if self.evaluation_functions:
        #     evaluation_metric_results = self.__run_evaluations(
        #         batch_prediction_result.predictions, ground_truth
        #     )

        #     eval_report_prompt = build_evaluation_prompt(
        #         current_prompt,
        #         self.task_description,
        #         evaluation_metric_results,
        #         batch_prediction_result,
        #         ground_truth,
        #         ClassificationReport.model_json_schema(),
        #     )
        #     # else:
        #     #     # attempt to use LLM to evaluate the ouput results
        #     eval_report = self.__evaluation_agent.run_sync(eval_report_prompt).output

        #     logger.info(f"Evaluation Result: {eval_report}")
        #     # Add the current prompt to the history before we modify
        #     self._policy_optimization_history.append(current_prompt)
        #     logger.info("Beginning prompt modification")
        #     modfication_prompt = build_optimizer_prompt(
        #         self._policy_optimization_history,
        #         current_prompt,
        #         eval_report.suggested_changes,
        #     )

        #     optimization_step_result = self.__modification_agent.run_sync(
        #         modfication_prompt
        #     ).output

        #     logger.debug(
        #         f"New version of Prompt\n=====\n{optimization_step_result.modified_prompt}\n=====\n"
        #     )

        #     current_prompt = optimization_step_result.modified_prompt
        #     logger.info(
        #         f"Epoch {epoch}: Finished round of optimization. \n Metrics: {evaluation_metric_results}"
        #     )
        #     epoch += 1

        # self._modified_prompt = current_prompt
        return max(__batch_prompt)

    def get_step_history(self):
        return self._policy_optimization_history

    def get_optimized_prompt(self):
        return self._modified_prompt

    def __store_history_to_file(self):
        pass


class GreedyOptimizer:
    pass


class BeamSearchOptimizer:
    pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
