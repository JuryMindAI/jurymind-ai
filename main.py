import os
import asyncio

# from jurymind.llm.openai import OpenAILLM
from dotenv import load_dotenv
from pydantic_ai import Agent
from jurymind.core.models import (
    OptimizationStepResult,
    OptimizationRunResult,
    JudgeDecision,
)

load_dotenv()


OPTIMIZER_INSTRUCTIONS = f"""Your job is to optimize a prompt from a user. You optimize 
by seeing how to rewrite, fix, or enhance the prompt to best work with an LLM and perform the required task.

Request to optimize:

{.model_json_schema()}

Output your results like so:

{OptimizationResult.model_json_schema()}

result:
"""
print(OPTIMIZER_INSTRUCTIONS)
agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=OptimizationResult,
    system_prompt=OPTIMIZER_INSTRUCTIONS,
    retries=3,
)

judge = Agent("openai:chatgpt-4o-mini")

curr_prompt = "Why do cat live do they happy life?"
prompt_hist = []
i = 0
max_iteration = 10


def optimize(
    optimization_request: PromptOptimizationRequest, max_iteration=5
) -> OptimizationResult:
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
