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
OPTIMIZER_TEMPLATE = """Your job is to optimize a prompt. You optimize 
by seeing how to rewrite, fix, or enhance the prompt to best work with an LLM and perform the task described below.

Below is the request to optimize format with field descriptions:

{task_desc}

Request to optimize values:

{optimize_job}

Output your results like so:

{output_schema}

"""

OPTIMIZER_DATA_GENERATOR = """ 
You are an expert AI agent which generates very challenging and unique examples based on a task description. 
Your must generate {n} extremely challenging, realistic, and very unique examples.
Be sure that you do not attempt to classify your own examples when creating this dataset.

Here is the task description:

{task_description}

Each example MUST adhere to the following rules exactly:

1. Each example must be realistic to the task description. 
2. The examples must be extremely challenging, and unique to previous examples.
3. The examples must be a challenge for even a powerful LLM to answer.
4. There must be an even number of positive and negative examples so we have a balanced dataset.
5. The examples must not include an explanation of the example.

Below is the request for data generation format with field descriptions:

{generator_job}

Optional examples to base generation off of:

{optional_examples}

You MUST not attempt to explain or classify the given task in your output. Only generate novel challenging examples based on the rules above and task description given.

You must output in the following structured format:

{output_schema}

result:
"""

CLASSIFICATION_INSTRUCTIONS = """
You perform classification on a batch of examples as defined in the prompt below. 
You must generate a list of predictions based on the prompts instructions

### Prompt: 

{prompt}

### Batch of examples to classify:
   
{batch}

### You must output your predictions in the following format:

{output_schema}

"""

EVALUATE_INSTRUCTIONS = """
Your job is to perform is generating a report on how well the given prompt was able to perform a task_description. 
You must take the predictions and compare those with the known ground_truth labels. 
You must then output suggested changes, the accuracy, and a confusion matrix. 

### Prompt:

{prompt}

### Task Description:
    
{task_description}

### Predictions:

{predictions}

### Ground truth labels:

{ground_truth}

###
Note that the ground-truth labels are __absolutely correct__, but the prompts (task description) may be incorrect and need modification.
###

You must format your report in this schema:

{output_schema}

"""


PROMPT_MODIFICATION = """

Agent is a large language model whose task is to modify a prompt based on the evaluation 
report from another agent. You must correct and modify the prompt based on the suggestions in the report.

### Prompt History ###

{prompt_history}

### Current Prompt ###

{current_prompt}

### Modification Suggestions ###

{suggestions}

###Instructions###

1. You will generate a new prompt based on the error analysis. 
2. Follow the analysis suggestions exactly and a predicted score for this prompt.
3. The new prompt must be different from all of the previous prompts.
4. The new prompt must be modified to prevent the failure cases.

You must follow the evaluation instructions exactly! Do not deviate from the suggestions, even if they seem opposite to what
you would do.

"""

# TODO: Probably put these elsewhere but for now keep here

def build_optimizer_prompt(task_desc, optimize_job, output_schema):
    return OPTIMIZER_TEMPLATE.format(
        task_desc=json.dumps(task_desc, indent=2),
        optimize_job=json.dumps(optimize_job, indent=2),
        output_schema=json.dumps(output_schema, indent=2),
    )


def build_generator_prompt(
    task_desc,
    generator_job,
    output_schema,
    optional_example="No Optional Examples for now",
    n=10
):
    return OPTIMIZER_DATA_GENERATOR.format(
        n=n,
        generator_job=json.dumps(task_desc, indent=2),
        task_description=json.dumps(generator_job, indent=2),
        optional_examples=optional_example,
        output_schema=json.dumps(output_schema, indent=2),
    )


def build_evaluation_prompt(
    prompt, task_description, batch_predictions, ground_truth, output_schema
):
    return EVALUATE_INSTRUCTIONS.format(
        n=len(batch_predictions.predictions),
        prompt=prompt,
        task_description=task_description,
        predictions=batch_predictions,
        ground_truth=ground_truth,
        output_schema=output_schema,
    )


def build_classifier_prompt(prompt, batch, output_schema):
    return CLASSIFICATION_INSTRUCTIONS.format(
        prompt=prompt, batch=batch, output_schema=output_schema
    )


def __build_optimizer_prompt(prompt_hist, curr_prompt, suggestions):
    return PROMPT_MODIFICATION.format(
        prompt_history=prompt_hist, current_prompt=curr_prompt, suggestions=suggestions
    )