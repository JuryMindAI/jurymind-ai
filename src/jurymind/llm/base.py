"""Base class for LLM integrations"""
from typing import Any, Self, Union
from abc import abstractmethod

class BaseLLM:
    
    def __init__(self, params=None):
        self.output_structure = None
        self.llm = None
        self.llm_params:dict = params
    
   
    @abstractmethod
    def __get_env_vars():
        """Function to be implemented to grab environment variables if they are available"""
        pass

    @abstractmethod
    def completion(self, prompt: Union[str]):
        """Function to call the LLM"""
        pass
    
    @abstractmethod
    def acompletion(self, prompt: Union[str]):
        """Async function to call the LLM if async is supported"""
        pass

    @abstractmethod
    def structured_output(self, llm: Self, structure) -> Self:
        """If an LLM supports structured output set it up here"""
        pass

    @abstractmethod
    def stream_completion(self, prompt: Union[str]):
        """streaming of llm response if supported by LLM"""
        pass
    
    @abstractmethod
    def astream_completion(self, prompt: Union[str]):
        """streaming of llm response if supported by LLM"""
        pass
