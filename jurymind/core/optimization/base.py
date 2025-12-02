"""
Classes and functions to run different opmtimization tasks.
"""

import json
import random
import uuid
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import tqdm
import mlflow
import numpy as np


from pprint import pprint
from typing import Callable
from functools import partial


from pydantic_ai import Agent
from loguru import logger

from mlflow.entities import SpanType
from pydantic import BaseModel

from jurymind.core.prompts.base import (
    build_classifier_prompt,
    build_evaluation_prompt,
    build_modification_prompt,
)

from jurymind.core.models import (
    BatchClassificationResult,
    ModificationReport,
    TaskExample,
    PromptVariants,
)
from itertools import repeat
from dataclasses import dataclass
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor


@dataclass
class BeamParent:
    prompt: str
    suggested_changes: Optional[str] = None
    evaluation_score: Optional[float] = None
    # optional: store history or other metadata
    history: Optional[List] = None


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
    Policy to Optimize prompts for a given task using beam search strategy.
    """

    def __init__(
        self,
        prompt: str,
        task_description: str,
        training_examples: list[TaskExample],
        model: str = "openai:gpt-4.1-mini",
        evaluator_model: str = "openai:gpt-4.1",  # Defaults to more advanced model for evaluations
        max_epochs: int = 5,
        num_workers: int = 2,
        search_type: str = "beam",  # greedy, beam defaults to beam search
        tracking_mlflow: bool = False,
        evaluation_examples: list[
            TaskExample
        ] = None,  # Optional list of evaluation examples to use during optimization. If none provided, training examples will be used to synthethise evaluation set.
        evaluators: list[Callable] = None,
        structured_output_type: BaseModel = None,
        return_global_max: bool = False,  # Evaluate all levels of search space for global max
        seed: int = 42,
    ):
        """
        Initialize prompt optimization

        Args:
            prompt (str): Prompt to optimize with this policy.
            task_description (str): Description of the task we are optimizing the prompt for.
            model (str, optional): LLM to use for optimizing the prompt. Defaults to "openai:gpt-4.1-mini".
            evaluator_model (str, optional): LLM to use for evaluating prompt changes. Defaults to "openai:gpt-4.1".
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
        self.tracking_mlflow: bool = tracking_mlflow
        # Setup the agents to be used in this policy workflow
        self.__classification_agent = Agent(
            self.agent_model_id, output_type=BatchClassificationResult, retries=3
        )
        self.__evaluation_agent = Agent(
            self.evaluator_model_id, output_type=ModificationReport, retries=3
        )

        self.__generation_agent = Agent(
            self.agent_model_id, output_type=PromptVariants, retries=3
        )

        self.__modification_agent = Agent(
            self.agent_model_id, output_type=PromptVariants, retries=3
        )
        # self.seed = seed

        if self.tracking_mlflow:
            logger.info("Setting up mlflow tracking for prompt optimization")
            logger.info(f"Experiment Name: OptimizationTag:{uuid.uuid4()}")
            logger.info("NOT IMPLEMENTED YET")
        #     mlflow.set_experiment(f"OptimizationTag:{uuid.uuid4()}")
        # self.__tracking_mlflow = tracking_mlflow

    def _candidate_generation(
        self, prompt: str, task_description: str, suggestions: str = None, n=5
    ) -> str:
        """
        Generates a list of candidates to explore for further optimization

        Args:
            prompt (str): prompt to expand from
            task_description (str): description of the task the prompt is trying to solve for
            suggestions: if suggestions are available from previous eval runs, provide them to the llm. Defaults to None.
        """
        raise NotImplementedError()

    def _select(self, prompts):
        """
        helper function to select the best candidates from a search

        Args:
            prompts (_type_): _description_
        """

        raise NotImplementedError()

    def __run_eval_funcs(
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
        self,
        prompt: str,
        batch_examples: list,
        batch_expectations: list,
        # sample_size=10,
    ) -> list[ModificationReport]:
        """
        Perform a search over the optimization space

        Args:
            prompt (str): prompt to explore the search space for
            examples (list): list of examples to use for evaluation
            sample_size (int, optional): number of examples to sample for evaluation. Defaults to 10.
        """

        # run each candidate in the space through the evaluator functions
        # once all candidates have run through eval functions, generate new modified variations off the top k scoring p_i-1 candidates
        beam_results = []
        logger.info(f"Working on Prompt: {prompt}")

        batch_prediction_prompt = build_classifier_prompt(
            prompt=prompt,
            batch=json.dumps(
                batch_examples
            ),  # dont give the model both the example and the labels, the llm may try to cheat.
        )

        logger.info(batch_prediction_prompt)

        batch_prediction_result = self.__classification_agent.run_sync(
            batch_prediction_prompt
        ).output

        logger.info("BATCH PREDICTION RESULTS")
        logger.info(batch_prediction_result.model_dump_json(indent=2))

        # list of evaluation results we need to merge with all the candidates
        evaluation_metric_results = self.__run_eval_funcs(
            batch_prediction_result.predictions, batch_expectations
        )

        eval_report_prompt = build_evaluation_prompt(
            prompt,
            self.task_description,
            evaluation_metric_results,
            batch_prediction_result,
            batch_expectations,
        )

        # attempt to use LLM to evaluate the ouput results
        eval_report = self.__evaluation_agent.run_sync(eval_report_prompt).output
        logger.info(f"Eval Report: \n\n{eval_report.model_dump_json(indent=2)}")

        # Return all the results from this layer of evaluation
        beam_results.append(eval_report)

        return beam_results

    def run(self, beam_width=5):
        """Performs beam search to help optimize the prompt"""
        all_beam_results: List = []

        # -------------------------------
        # START WITH SINGLE PARENT
        # -------------------------------
        parents = [BeamParent(prompt=self.original_prompt)]

        # -------------------------------
        # HELPER: Generate children from parents
        # -------------------------------
        def generate_children(parents: List[BeamParent], n: int) -> List[str]:
            children = []
            for parent in parents:
                modification_prompt = build_modification_prompt(
                    all_beam_results,
                    parent.prompt,
                    parent.suggested_changes,
                    n=n,
                )
                variants = self.__modification_agent.run_sync(
                    modification_prompt
                ).output.variants
                children.extend(variants)
            return children

        # -------------------------------
        # MAIN BEAM SEARCH LOOP
        # -------------------------------
        for epoch in range(self.max_epochs):
            logger.info(f"Running Epoch: {epoch}")

            # 1 Generate children
            children_prompts = generate_children(parents, n=beam_width)

            examples = [x.example for x in self.training_examples]
            expectations = [x.expectation for x in self.training_examples]
            x = []

            logger.info(len(examples))

            # 2 Search Space
            training_sample, _, expectation_sample, _ = train_test_split(
                examples,
                expectations,
                train_size=100,  # or 0.5, 1000, etc.
                stratify=expectations,
            )

            logger.info(len(training_sample))
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                # Kick off each partial
                search_fn = partial(self.__search_space)
                child_results = list(
                    pool.map(
                        search_fn,
                        children_prompts,
                        repeat(training_sample),
                        repeat(expectation_sample),
                    )
                )

            child_results = np.array(child_results).flatten()
            logger.info(f"child_results: \n {child_results}")
            # 3️ Sort and prune to top-K
            top_k_results = child_results[:beam_width]

            # 4️ Convert to BeamParent objects for next epoch
            parents = [
                BeamParent(
                    prompt=r.original_prompt,
                    suggested_changes=r.suggested_changes,
                    evaluation_score=r.accuracy,
                )
                for r in top_k_results
            ]

            # 5️ Add to global history
            all_beam_results.extend(top_k_results)

        # Perform final evaluation across all beam results to pick the best prompt
        return all_beam_results  # take arg max

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
