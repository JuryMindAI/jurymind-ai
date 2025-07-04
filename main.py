import os
import asyncio
# from jurymind.llm.openai import OpenAILLM
from dotenv import load_dotenv
from pydantic_ai import Agent
from jurymind.agents.models import OptimizationResult, OptimizationRequest

load_dotenv()


OPTIMIZER_INSTRUCTIONS = "Your job is to optimize a prompt from a user. You optimize by seeing how to rewrite, fix, or enhance the prompt to best work with an LLM."

agent = Agent("openai:gpt-4o", output_type=OptimizationResult, system_prompt=OPTIMIZER_INSTRUCTIONS, retries=3)

curr_prompt = "What do cat live so short do they live happy life?"
prompt_hist = []
i = 0
while i<10:
    # call agent, get response and see if we should keep optimizing or not
    result = agent.run_sync(curr_prompt)
    print(result)
    # if result.stop:
    #     print("Stopping")
    #     break
    i += 10
    
print(curr_prompt) # Outputs the response