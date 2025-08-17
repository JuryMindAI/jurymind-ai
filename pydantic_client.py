from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from dotenv import load_dotenv

load_dotenv()
server = MCPServerStreamableHTTP("http://localhost:8000/mcp")
agent = Agent('openai:chatgpt-4o-latest', toolsets=[server])

async def main():
    async with agent:  
        result = await agent.run('What is the weather?')
    print(result.output)