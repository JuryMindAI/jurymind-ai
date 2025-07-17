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
Be sure that you do not attempt to classify your own examples when creating this dataset.

Each example MUST adhere to the following rules exactly:

1. Each example must be realistic to the task description. 
2. The examples must be extremely challenging, and unique to previous examples.
3. The examples must be a challenge for even a powerful LLM to answer.
4. There must be an even number of positive and negative examples so we have a balanced dataset.
5. The examples must not include an explanation of the example.

Here is the task description:

{task_description}

Below is the request for data generation format with field descriptions:

{generator_job}

Optional examples to base generation off of:

{optional_examples}

You must output in the following structured format:

{output_schema}

You MUST not attempt to explain or classify the given task in your output. Only generate novel challenging examples based on the rules above and task description given.
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
You are an expert prompt evaluation AI agent that must evaluate the result of a batch of {n} predictions
for a prompt on the given task description.

Prompt:

{prompt}

Task Description:
    
{task_description}

Batch to evaluate:

{batch_predictions}

Ground truth:

{ground_truth}

Instructions are as follows:

1. You must precisely evaluate the batch taking care to note where the prompt could be improved.
2. You must generate an accuracy score for the batch given the prediction and compare it to the ground truth label
3. You must generate a confusion matrix based on the batch and ground truth labels
4. You must come up with a series of suggested changes that will meaningfully improve the prompt.


Structure the output like so:

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

    def optimize(self, prompt):
        pass


class DataGenerationPolicy:
    pass


class LLMEvaluationPolicy:
    pass
