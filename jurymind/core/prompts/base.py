from abc import abstractmethod


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant whose role is 
    to carry out the request from the user to the best of your ability."""


class BasePromptTemplate:

    def __init__():
        pass

    @abstractmethod
    def format(self):
        pass
