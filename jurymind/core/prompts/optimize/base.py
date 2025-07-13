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

{output_schema}

result:
"""

OPTIMIZER_DATA_GENERATOR = """ 
You are an expert AI agent which generates very challenging and unique examples based on the given task description. 
Your must generate {n} extremely challenging, realistic, and very different examples.
Be sure that you do not accidentally attempt to classify your own generated examples.

Each example must also adhere to the following rules exactly:

1. Each example must be realistic to the task description. 
2. The examples must be extremely challenging, and unique to previous examples.
5. There must be an even number of positive and negative examples so we have a balanced dataset.
6. The examples must not include an explanation of the example.

Here is the task description:

{task_description}

Below is the request for data generation format with field descriptions:

{generator_job}


You must output in the following structured format:

{output_schema}

result:
"""

CLASSIFICATION_INSTRUCTIONS = """
You're task is to classify the following examples based on the prompts instructions. 
Assume this is a binary classification task. You will classify the examples according to the prompt
and generate the accuracy, confusion matrix, suggestions as to how to modify the prompt and why it failed, and list of failure cases
from the dataset.

Task prompt: 

{prompt}

Batch to classify using the above task prompt:
   
{batch}

You must output your results to the following format:

{output_schema}

Report:
"""

EVALUATE_INSTRUCTIONS = """
You are an expert prompt evaluation AI agent that must evaluate if the prompt

Prompt:

Accuracy: 
{}

Confusion Matrix: 
{}

Task Description:
    
{task_description}

Instructions are as follows:
1. Perform the task on the datapoint
2. Generate a prediction based on the task. If the task doesnt not explain how to label the prediction, assume binary labels.
3. Analyise the prompt to the datapoint and come up with a reason why this prompt may and may not work at predicting the label.


Structure the output like so:
{classification_result}
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

    def optimize(self, prompt):
        pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
