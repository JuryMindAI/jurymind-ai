import json
from jurymind.core.models import OptimizationStepResult

OPTIMZE_PROMPT_STEP = """You are an expert at optimizing prompts for a given task."""

OPTIMIZER_INSTRUCTIONS = f"""Your job is to optimize a prompt from a user. You optimize 
by seeing how to rewrite, fix, or enhance the prompt to best work with an LLM and perform the required task.

Below is the request to optimize format with field descriptions:

{{task_desc}}

Request to optimize values:

{{optimize_job}}

Output your results like so:

{OptimizationStepResult.model_json_schema()}

result:
"""

# only parameterize this part
OPTIMIZER_TEMPLATE = """Your job is to optimize a prompt from a user. You optimize 
by seeing how to rewrite, fix, or enhance the prompt to best work with an LLM and perform the required task.

Below is the request to optimize format with field descriptions:

{task_desc}

Request to optimize values:

{optimize_job}

Output your results like so:

{schema}

result:
"""

OPTIMIZER_DATA_GENERATOR = """ 
You are an expert AI system which generates data sets about the given task description. Your task is to generate {n} extremely challenging, realistic, and very different examples based on the task description provided.
Each generated example must be 3 sentences long. Be sure that you do not accidentally attempt to classify your own generated example.

The each exmaple must also adhere to the following:

1. The example must be extremely challenging, unique to previous examples.
2. The example must be an even number of positive and negative datapoints so we have a balanced dataset.
3. The example must not include an explanation of the example.

Below is the request for data generation format with field descriptions:

{task_desc}

Here is the task description:

{generator_job}

You must output in the following structured format:

{output_schema}

result:
"""


class PromptOptimizationPolicy:
    """
    Optimization Policy for tuning prompts to a given task.
    """

    def __init__(self, optimization_job, model="", max_epochs=1, num_workers=1):
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.optimization_result = None
        self.optimization_request = None
        self.search_type = None  # grid, random, beam

    def __build_optimizer_prompt(self, task_desc, optimize_job, schema):
        return OPTIMIZER_TEMPLATE.format(
            task_desc=json.dumps(task_desc, indent=2),
            optimize_job=json.dumps(optimize_job, indent=2),
            schema=json.dumps(schema, indent=2),
        )

    def optimize(self):
        pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
