"""
Classes and functions to run different opmtimization tasks.
"""

import json
import random
from pprint import pprint
from typing import Callable

from pydantic_ai import Agent
from loguru import logger
import mlflow

from mlflow.entities import SpanType
import tqdm
from pydantic import BaseModel

from jurymind.core.prompts.base import (
    build_classifier_prompt,
    build_evaluation_prompt,
    build_modification_prompt,
)

from jurymind.core.models import (
    BatchClassificationResult,
    ModificationReport,
    OptimizationStepResult,
    TaskExample,
    PromptVariants,
)

from dataclasses import dataclass
from typing import Optional, List


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
    Optimize a prompt to a specific task.
    """

    def __init__(
        self,
        prompt: str,
        task_description: str,
        model: str = "openai:gpt-4.1-mini",
        evaluator_model: str = "openai:gpt-4.1",  # Defaults to more advanced model for evaluations
        max_epochs: int = 10,
        num_workers: int = 1,
        search_type: str = "beam",
        tracking_mlflow: bool = False,
        training_examples: list[TaskExample] = None,
        evaluation_examples: list[TaskExample] = None,
        evaluators: list[Callable] = None,
        structured_output_type: BaseModel = None,
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
            self.evaluator_model_id, output_type=ModificationReport, retries=3
        )

        self.__generation_agent = Agent(
            self.agent_model_id, output_type=PromptVariants, retries=3
        )

        self.__modification_agent = Agent(
            self.agent_model_id, output_type=PromptVariants, retries=3
        )

        # self.__tracking_mlflow = tracking_mlflow

    def _candidate_generation(
        self, prompt, task_description, suggestions=None, n=5
    ) -> str:
        """
        Generates a list of candidates to explore for further optimization

        Args:
            prompt (_type_): prompt to expand from
            task_description (_type_): description of the task the prompt is trying to solve for
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

    def __run_evaluations(
        self, model_predictions: list[str], data_expectations: list[str]
    ):
        """
        Runs the evaluation functions, if provided, over the evaluation examples to
        align the prompt changes to the target function.
        """
        results = []
        for func in self.evaluation_functions:
            logger.info(func)
            results.append(func(model_predictions, data_expectations))
        return results

    def __search_space(
        self, search_space: list, examples: list, expectations=None, sample_size=10, k=5
    ) -> list[ModificationReport]:
        """
        Perform a search over the space of prompts to optimize for the given task

        Args:
            space (_type_): Search space of prompts to run against example data

        """
        # run each candidate in the space through the evaluator functions
        # once all candidates have run through eval functions, generate new modified variations off the top k scoring p_i-1 candidates
        depth_results = []
        for prompt in search_space:

            # TODO: Should multi thread this since each prompt would be its own set of work
            minibatch_sample = random.sample(examples, sample_size)

            batch_prediction_prompt = build_classifier_prompt(
                prompt=prompt,
                batch=json.dumps(
                    minibatch_sample
                ),  # dont give the model both the example and the labels, the llm may try to cheat.
            )

            batch_prediction_result = self.__classification_agent.run_sync(
                batch_prediction_prompt
            ).output

            logger.info(f"Batch Prediction Results: {batch_prediction_result}")
            # list of evaluation results we need to merge with all the candidates
            evaluation_metric_results = self.__run_evaluations(
                batch_prediction_result.predictions, expectations
            )

            logger.info(f"Evaluation Metric Results: {evaluation_metric_results}")
            eval_report_prompt = build_evaluation_prompt(
                prompt,
                self.task_description,
                evaluation_metric_results,
                batch_prediction_result,
                expectations,
            )

            # attempt to use LLM to evaluate the ouput results
            eval_report = self.__evaluation_agent.run_sync(eval_report_prompt).output
            logger.info(f"Eval Report: {eval_report}")
            # Return all the results from this layer of evaluation
            depth_results.append(eval_report)

        return depth_results

    def run(self, beam_width=5):
        """Performs beam search to help optimize prompt"""
        all_beam_results: List = []

        examples = [x.example for x in self.evaluation_examples]
        expectations = [x.label for x in self.evaluation_examples]

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

            # 1️⃣ Generate children
            children_prompts = generate_children(parents, n=beam_width)

            # 2️⃣ Evaluate children
            child_results = self.__search_space(
                children_prompts, examples, expectations
            )

            # 3️⃣ Sort and prune top-K
            child_results_sorted = sorted(
                child_results, key=lambda x: x.accuracy, reverse=True
            )
            top_k_results = child_results_sorted[:beam_width]

            # 4️⃣ Convert to BeamParent objects for next epoch
            parents = [
                BeamParent(
                    prompt=r.original_prompt,
                    suggested_changes=r.suggested_changes,
                    evaluation_score=r.accuracy,
                )
                for r in top_k_results
            ]

            # 5️⃣ Add to global history
            all_beam_results.extend(top_k_results)

        return all_beam_results

    # def run(self, beam_width=5):
    #     """Run the optimization steps for this policy. Uses Beam search to find optimal search space."""
    #     # runs the workflow for this policy
    #     epoch = 0
    #     # each step holds the current prompt
    #     # Generate k variants up front to get and initial search space beyond a singular prompt
    #     __p0_variants = self.__generation_agent.run_sync(self.original_prompt).output
    #     # __all_candi = __p0_variants.variants + [self.original_prompt]

    #     beam_candidates = __p0_variants.variants + [self.original_prompt]
    #     all_beam_results = []

    #     examples = [x.example for x in self.evaluation_examples]
    #     expectations = [x.label for x in self.evaluation_examples]

    #     # pbar = tqdm.tqdm(desc="Prompt Optimizing", total=self.max_epochs)
    #     # Loop for n epochs, collecting up the results per pass
    #     while epoch < self.max_epochs:
    #         logger.info(f"Running Epoch: {epoch}")
    #         # search the current depth of space
    #         beam_results = self.__search_space(beam_candidates, examples, expectations)

    #         logger.info(f"Beam Result: {beam_results}")
    #         # get top k from beam candidates (list of candidate with eval score)

    #         top_k = sorted(beam_results, key=lambda x: x.accuracy, reverse=True)[
    #             :beam_width
    #         ]
    #         # add the top k results to all the beam search results
    #         all_beam_results.extend(top_k)

    #         # reset beam candidates for next round of searching
    #         beam_candidates = []

    #         children = []
    #         for prompt in top_k:
    #             children.extend(
    #                 self.__modification_agent.run_sync(
    #                     build_modification_prompt(
    #                         all_beam_results,  # current history of top_k from each beam
    #                         prompt.original_prompt,
    #                         prompt.suggested_changes,
    #                         n=beam_width,
    #                     )
    #                 ).output.variants
    #             )

    #             # now generate next round of candidates based off of the top k we just got from above
    #             # for prompt in top_k:
    #             # modification_prompt = build_modification_prompt(
    #             #     all_beam_results,  # current history of top_k from each beam
    #             #     prompt.original_prompt,
    #             #     prompt.suggested_changes,
    #             #     n=beam_width,
    #             # )
    #         #     # do I want to keep ALL candidates or just the top ones from last round
    #         #     beam_candidates.extend(
    #         #         self.__modification_agent.run_sync(
    #         #             modification_prompt
    #         #         ).output.variants
    #         #     )
    #         epoch += 1
    #         # pbar.update(1)

    #     return all_beam_results

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
