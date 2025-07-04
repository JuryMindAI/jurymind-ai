from pydantic_ai import Agent, RunContext  
from pydantic import BaseModel


class BaseAgent:
    def __init__(self, model_id, system_prompt, ):
        self.agent = Agent()


class JuryAgent:
    pass


class JudgeAgent:
    pass


class DefenseAgent:
    pass


class ProsecutorAgent:
    pass

class OptimizerAgent(BaseAgent):
    pass