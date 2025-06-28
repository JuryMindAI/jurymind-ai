"""Base class for LLM integrations"""
from typing import Self, Union
from abc import abstractmethod


class BaseLLM:
    """Base LLM class that all LLM classes must implement"""
    def __init__(self, llm, params=None):
        self.__output_structure = None
        self.llm: BaseLLM = llm
        self.llm_params: dict = params

    @abstractmethod
    def completion(self, prompt: Union[str]):
        """Function to call the LLMs completion mechanism"""
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
    def __stream_completion(self, event_stream):
        """streaming of llm response if supported by LLM"""
        pass
    
    @abstractmethod
    def astream_completion(self, prompt: Union[str]):
        """streaming of llm response if supported by LLM"""
        pass
