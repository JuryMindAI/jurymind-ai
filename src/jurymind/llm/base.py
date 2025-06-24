"""Base class for LLM integrations"""
from typing import Any, Self, Union
from abc import abstractmethod
class BaseLLM:
    
    def __init__(self):
        self.output_structure = None

    @abstractmethod
    def completion(self, prompt: Union[str]):
        pass
    
    @abstractmethod
    def acompletion(self, prompt: Union[str]):
        pass

    @abstractmethod
    def structured_output(self, llm: Self, structure) -> Self:
        """If an LLM supports structured output set it up here"""
        pass
