import os
import asyncio
import json

# from jurymind.llm.openai import OpenAILLM
from dotenv import load_dotenv
from pydantic_ai import Agent
from jurymind.core.models import (
    OptimizationStepResult,
    OptimizationRunResult,
    PromptOptimizationRequest,
    DataGenerationOutput,
    ClassificationReport,
    BatchClassificationResult,
    OptimizationStepResult,
)

from jurymind.core.prompts.optimize.base import (
    OPTIMIZER_TEMPLATE,
    OPTIMIZER_DATA_GENERATOR,
    CLASSIFICATION_INSTRUCTIONS,
    EVALUATE_INSTRUCTIONS,
    OPTIMIZER_INSTRUCTIONS,
    PROMPT_MODIFICATION,
)

load_dotenv()

agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=OptimizationStepResult,
    retries=3,
)

generator_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=DataGenerationOutput,
    retries=3,
)

classification_agent = Agent(
    "openai:gpt-4.1-mini", output_type=BatchClassificationResult, retries=3
)

evaluation_agent = Agent("openai:gpt-4o", output_type=ClassificationReport, retries=3)

modification_agent = Agent(
    "openai:gpt-4.1", output_type=OptimizationStepResult, retries=3
)

# judge = Agent("openai:chatgpt-4.1-mini")

curr_prompt = "Why do cat live do they happy life?"
prompt_hist = []
i = 0
max_iteration = 10


def __build_optimizer_prompt(task_desc, optimize_job, output_schema):
    return OPTIMIZER_TEMPLATE.format(
        task_desc=json.dumps(task_desc, indent=2),
        optimize_job=json.dumps(optimize_job, indent=2),
        output_schema=json.dumps(output_schema, indent=2),
    )


def __build_generator_prompt(
    task_desc,
    generator_job,
    output_schema,
    optional_example="No Optional Examples for now",
):
    return OPTIMIZER_DATA_GENERATOR.format(
        n=10,
        generator_job=json.dumps(task_desc, indent=2),
        task_description=json.dumps(generator_job, indent=2),
        optional_examples=optional_example,
        output_schema=json.dumps(output_schema, indent=2),
    )


def __build_evaluation_prompt(
    prompt, task_description, batch_predictions, ground_truth, output_schema
):
    return EVALUATE_INSTRUCTIONS.format(
        n=len(batch_predictions.predictions),
        prompt=prompt,
        task_description=task_description,
        predictions=batch_predictions,
        ground_truth=ground_truth,
        output_schema=output_schema,
    )


def __build_classifier_prompt(prompt, batch, output_schema):
    return CLASSIFICATION_INSTRUCTIONS.format(
        prompt=prompt, batch=batch, output_schema=output_schema
    )


def __build_optimizer_prompt(prompt_hist, curr_prompt, suggestions):
    return PROMPT_MODIFICATION.format(
        prompt_history=prompt_hist, current_prompt=curr_prompt, suggestions=suggestions
    )


def optimize(
    optimization_request: PromptOptimizationRequest, max_iteration=5
) -> OptimizationRunResult:

    # sys_prompt = __build_optimizer_prompt(
    #     task_desc=json.dumps(PromptOptimizationRequest.model_json_schema(), indent=2),
    #     optimize_job=json.dumps(optimization_request.model_dump_json(), indent=2),
    #     output_schema=json.dumps(OptimizationStepResult.model_json_schema(), indent=2),
    # )

    # generator_prompt = __build_generator_prompt(
    #     task_desc=PromptOptimizationRequest.model_json_schema(),
    #     generator_job=optimization_request.model_dump_json(),
    #     output_schema=DataGenerationOutput.model_json_schema(),
    # )

    with open("small_data.json", "r") as f:
        dataset = json.load(f)

    print(len(dataset))

    """
        TODO:
        1. create the optimization system prompt Done
        2. Take the task description and prompt and generate examples within the task. Done
        3. Pull a subset of generated samples and have LLM label them, or allow for human input examples. Done
        4. Search the space for the best optimized prompt (treat this like a classification problem)
            1. Take prompt, apply it to the generated samples
            2. evalute the prompts ability to elicit correct behavior
            3. create error report to send to judge agents
            4. generate additional sample prompts based on the error report findings and go to step 1.
        5. return the optimized prompt
    """
    curr_prompt = optimization_request.prompt

    i = 0
    while i < max_iteration:
        print(f"Iteration: {i+1}")
        # call agent, get response and see if we should keep optimizing or not
        # result = agent.run_sync(curr_prompt)
        # gen_examples = generator_agent.run_sync(generator_prompt).output
        # print(gen_examples.examples)

        examples = [x["review"] for x in dataset]
        ground_truth = [x["label"] for x in dataset]

        batch_prediction_prompt = __build_classifier_prompt(
            prompt=optimization_request.prompt,
            batch=json.dumps(
                examples
            ),  # dont give the model both the example and the labels, I think the AI will cheat
            output_schema=BatchClassificationResult.model_json_schema(),
        )

        batch_prediction_result = classification_agent.run_sync(
            batch_prediction_prompt
        ).output

        eval_prompt = __build_evaluation_prompt(
            optimization_request.prompt,
            optimization_request.task_description,
            batch_prediction_result,
            ground_truth,
            ClassificationReport.model_json_schema(),
        )

        eval_result = evaluation_agent.run_sync(eval_prompt).output

        prompt_hist.append(curr_prompt)

        modfication_prompt = __build_optimizer_prompt(
            prompt_hist,
            curr_prompt,
            eval_result.suggested_changes,
        )
        
        optimization_step_result = modification_agent.run_sync(modfication_prompt).output
        
        curr_prompt = optimization_step_result.modified_prompt
        i+=1
        print(f"New version of prompt: {curr_prompt}")
        print(f"Explanation for the changes: {optimization_step_result.explanation_of_changes}")

    return prompt_hist, curr_prompt, optimization_step_result.explanation_of_changes


# print(
#     PromptOptimizationRequest(
#         task_description="AHHH", prompt="hello", examples={}
#     ).model_json_schema()
# )
# print()
# print(
#     PromptOptimizationRequest(
#         task_description="AHHH", prompt="hello", examples={}
#     ).model_dump_json()
# )

history, optimized_prompt, explanation = optimize(
    PromptOptimizationRequest(
        task_description="The task is a binary classification task to check if a review has spoilers in them or not.",
        prompt="Do these movie reviews contain spoilers? You answer with a True or False.",
        iterations=10
    )
)

print("### Optimized Prmpt###")
print(optimized_prompt)
print()
print("###Optimization steps####")
print(history)
print()
print("###Explanation for change###")
print(explanation)
