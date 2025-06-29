from llm.base import BaseLLM
import asyncio
import os

from openai import OpenAI
from jurymind.core.prompts import 
class OpenAiLLM(BaseLLM):
    """OpenAI api llm completion support"""

    def __init__(self, api_key=None, model="gpt-4", params=None):

        if not api_key:
            api_key = os.getnv("OPENAI_API_KEY")

        self.super(OpenAI(api_key=api_key), params=params)

    def completion(self, prompt, stream=False):

        if not prompt:
            raise RuntimeError("Prompt cannot be None or empty String.")

        message = self.__format_message(prompt)
        response = self.llm.chat.completions.create(message, stream=stream)

        if not stream:
            """Return response object"""
            return response

        return self.__stream_completion(response)

        # begin streaming the response back

    def __format_message(user_prompt, system_prompt=None):

        message = [
            {
                "role": "system",
                "content": (
                    DEFAULT_SYSTEM_PROMPT if not system_prompt else system_prompt
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        return message

    async def acompletion(self, prompt):
        return NotImplemented

    def __stream_completion(self, event_stream):
        if not event_stream:
            raise RuntimeError("event stream cannot be None")
        for chunk in event_stream:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content

    async def astream_completion(self, prompt):
        return NotImplemented
