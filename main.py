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
    DataGenerationOutput
)

from jurymind.core.prompts.optimize.base import (
    OPTIMIZER_INSTRUCTIONS,
    OPTIMZE_PROMPT_STEP,
    OPTIMIZER_TEMPLATE,
    OPTIMIZER_DATA_GENERATOR
)

load_dotenv()

agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=OptimizationStepResult,
    # system_prompt=OPTIMIZER_INSTRUCTIONS,
    retries=3,
)

generator_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=DataGenerationOutput,
    # system_prompt=OPTIMIZER_INSTRUCTIONS,
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
    
    
def __build_generator_prompt(task_desc, generator_job, output_schema):
    return OPTIMIZER_DATA_GENERATOR.format(
        n = 10,
        task_desc=json.dumps(task_desc, indent=2),
        generator_job=json.dumps(generator_job, indent=2),
        output_schema=json.dumps(output_schema, indent=2),
    )


def optimize(
    optimization_request: PromptOptimizationRequest, max_iteration=5
) -> OptimizationRunResult:

    sys_prompt = __build_optimizer_prompt(
        task_desc=json.dumps(PromptOptimizationRequest.model_json_schema(), indent=2),
        optimize_job=json.dumps(optimization_request.model_dump_json(), indent=2),
        schema=json.dumps(OptimizationStepResult.model_json_schema(), indent=2),
        )
    
    generator_prompt = __build_generator_prompt(        
        task_desc=PromptOptimizationRequest.model_json_schema(),
        generator_job=optimization_request.model_dump_json(),
        output_schema=DataGenerationOutput.model_json_schema(),
        )
   
    print(generator_prompt)
    """
        TODO:
        1. create the optimization system prompt
        2. Take the task description and prompt and generate examples within the task.
        3. Pull a subset of generated samples and have LLM label them, or allow for human input examples.
        4. Search the space for the best optimized prompt (treat this like a classification problem)
            1. Take prompt, apply it to the generated samples
            2. evalute the prompts ability to elicit correct behavior
            3. create error report to send to judge agents
            4. generate additional sample prompts based on the error report findings and go to 1.
        5. return the optimized prompt
    """
    curr_prompt = sys_prompt
    i = 0
    while i < max_iteration:
        print(f"Iteration: {i+1}")
        # call agent, get response and see if we should keep optimizing or not
        result = agent.run_sync(curr_prompt)
        gen_result = generator_agent.run_sync(generator_prompt)
        curr_prompt = result.output.optimized_prompt
        prompt_hist.append(result)
        # if result.stop:
        #     print("Stopping")
        #     break
        if result.output.stop:
            print("Stopping iteration")
            break
        i += 1
        
    return result, gen_result


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

result, gen_result = optimize(PromptOptimizationRequest(task_description="The task is classification task to check if movie reviews have spoilers in them.", prompt="Do these movie reviews contain spoilers? Response with a yes or no."))

print(result)
print()
print(gen_result)