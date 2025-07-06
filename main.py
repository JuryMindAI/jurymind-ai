import os
import asyncio

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
    OPTIMIZER_TEMPLATE
)

load_dotenv()

agent = Agent(
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


def optimize(
    optimization_request: PromptOptimizationRequest, max_iteration=5
) -> OptimizationRunResult:

    curr_prompt = OPTIMIZER_INSTRUCTIONS.format()
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


print(PromptOptimizationRequest(task_description="AHHH", prompt="hello", examples={}).model_json_schema())
print()
print(PromptOptimizationRequest(task_description="AHHH", prompt="hello", examples={}).model_dump_json())

print(OPTIMIZER_TEMPLATE)