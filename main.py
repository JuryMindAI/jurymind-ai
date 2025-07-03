import os
# from jurymind.llm.openai import OpenAILLM
from dotenv import load_dotenv

load_dotenv()


from pydantic_ai import Agent
agent = Agent("openai:gpt-4", system_prompt="Be a helpful assistant.")
result = await agent.run("Hello, how are you?")
print(result.data)  # Outputs the response