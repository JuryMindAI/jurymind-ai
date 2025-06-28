from llm.base import BaseLLM
import asyncio
import os

from openai import OpenAI


class OpenAiLLM(BaseLLM):
    """OpenAI api llm completion support"""
    def __init__(self, api_key=None, model="gpt-4", params=None):
          
        if not api_key:
            api_key = os.getnv("OPENAI_API_KEY")

        self.super(OpenAI(api_key=api_key), params=params)

    def completion(self, prompt, stream=False):
        message = self.__format_message(prompt)
        response = self.llm.chat.completions.create(message, stream=stream)
        
        if not stream:
            return response
        
        # begin streaming the response back
    
    def __format_message(user_prompt):
        message = [
            {"role": "system"},
            {"role": "user", }
        ]
        return message
    
    async def acompletion(self, prompt):
        return NotImplemented
    
    def stream_completion(self, prompt):
        return 
    
    async def astream_completion(self, prompt):
        return NotImplemented
        
    
        