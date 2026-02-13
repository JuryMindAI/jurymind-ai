import inspect

from functools import wraps


class Evaluator:
    pass


class CustomEvaluator(Evaluator):
    pass


def evaluator(func):
    """Decorator function to help make it easier to add custom evalautions to the optimization process"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function's signature
        sig = inspect.signature(func)
        # Count the number of parameters in the function
        num_params = len(sig.parameters)

        if num_params != 2:
            raise TypeError(
                f"Function '{func.__name__}' must have exactly 2 parameters, but it has {num_params}."
            )

        # Call the original function
        return func(*args, **kwargs)

    return wrapper
