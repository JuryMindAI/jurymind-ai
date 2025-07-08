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
)

from jurymind.core.prompts.optimize.base import (
    OPTIMIZER_INSTRUCTIONS,
    OPTIMZE_PROMPT_STEP,
    OPTIMIZER_TEMPLATE,
)

load_dotenv()

agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=OptimizationStepResult,
    system_prompt=OPTIMIZER_INSTRUCTIONS,
    retries=3,
)

generator_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=OptimizationStepResult,
    system_prompt=OPTIMIZER_INSTRUCTIONS,
    retries=3,
)

judge = Agent("openai:chatgpt-4.1-mini")

curr_prompt = "Why do cat live do they happy life?"
prompt_hist = []
i = 0
max_iteration = 10


def __build_optimizer_prompt(task_desc, optimize_job, schema):
    return OPTIMIZER_TEMPLATE.format(
        task_desc=json.dumps(task_desc, indent=2),
        optimize_job=json.dumps(optimize_job, indent=2),
        schema=json.dumps(schema, indent=2),
    )


def optimize(
    optimization_request: PromptOptimizationRequest, max_iteration=5
) -> OptimizationRunResult:

    sys_prompt = __build_optimizer_prompt(
        task_desc=json.dumps(PromptOptimizationRequest.model_json_schema(), indent=2),
        optimize_job=json.dumps(optimization_request.model_dump_json(), indent=2),
        schema=json.dumps(OptimizationStepResult.model_json_schema(), indent=2),
        )
   
   
   """
   TODO:
   1. create the optimization system prompt
   2. Take the task description and prompt and generate examples within the task.
   3. Pull a subset of generated samples and have LLM label them, or allow for human input examples.
   4. Search the space for the best optimized prompt (treat this like a classification problem)
   5. return the optimized prompt
   """
    curr_prompt = sys_prompt
    i = 0
    while i < max_iteration:
        print(f"Iteration: {i+1}")
        # call agent, get response and see if we should keep optimizing or not
        result = agent.run_sync(curr_prompt)
        curr_prompt = result.output.optimized_prompt
        prompt_hist.append(result)
        # if result.stop:
        #     print("Stopping")
        #     break
        if result.output.stop:
            print("Stopping iteration")
            break
        i += 1
        
    return result


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

result = optimize(PromptOptimizationRequest(task_description="The task is to check if movie reviews have spoilers in them.", prompt="Do these movie reviews contain spoilers? Response with a yes or no."))

print(result)
print(type(result))