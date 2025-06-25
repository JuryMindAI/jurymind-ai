from llm.base import BaseLLM
from abc import abstractmethod
import os

import openai

class OpenAI(BaseLLM):
    """OpenAI api support. This support is only for the LLM API calls."""
    def __init__(self, api_key=None, model="gpt-4", params=None, ):
        
        self.super(params=params)
        self.llm = OpenAI()

    def completion(self, prompt):
        return super().completion(prompt)
    
    def acompletion(self, prompt):
        return super().acompletion(prompt)
    
    def stream_completion(self, prompt):
        return super().stream_completion(prompt)
    
    def astream_completion(self, prompt):
        return super().astream_completion(prompt)
    
        