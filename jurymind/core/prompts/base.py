import json
from jurymind.core.models import OptimizationStepResult

OPTIMZE_PROMPT_STEP = """You are an expert at optimizing prompts for a given task."""

OPTIMIZER_INSTRUCTIONS = f"""Your job is to optimize a prompt for a specific task which is described below. You optimize by rewriting, editing 
and or enhance the prompt to best work with an LLM and perform the required task.

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

### Below is the request to optimize format with field descriptions: ###

{task_desc}

### Request to optimize values: ###

{optimize_job}

### Suggestions for changes to the prompt ###

{suggestions}

"""

PROMPT_VARIANT_GENERATOR_INST = """
You are an expert AI Agent which generates variants of a given prompt
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

PromptVariants:
"""

CLASSIFICATION_INSTRUCTIONS = """
You perform classification on a batch of examples as defined in the prompt. 
You must generate a list of predictions based on the prompts instructions. Return the predictions in the same order of the batch.


### Prompt: ### 

{prompt}

### Batch of examples to classify: ###
   
{batch}

ClassificationResult:
"""

EVALUATE_INSTRUCTIONS = """
Your job is generating a report on how well the given prompt was able to perform a task_description. 
You must take the predictions and compare those with the known ground truth labels. 
You must then output suggested changes to be made to the prompt that will help improve the metrics. You must also give an explanation as to why these changes
will improve the scores. DO NOT OVERFIT TO THE EXAMPLES. If there is concern for overfitting, mention that in your explanation.

### Prompt:

{prompt}

### Task Description:
    
{task_description}

### Predictions:

{predictions}

### Evaluation Metric Results:

{metric_results}

### Ground truth labels:

{ground_truth}

###
Note that the ground-truth labels are __absolutely correct__, but the prompts may be incorrect and need modification.
###

"""

META_EVAL_RESULTS = """
Your task is to look at a series of examples and metrics that were calculated with the current prompt and output suggestions to apply
to a modified version of the prompt that will improve the metrics on this task.
"""

PROMPT_MODIFICATION = """

You are an AI agent whose task is to modify a prompt based on the following suggestions, if provided, from a more advanced AI You should generate {n} variants of the current prompt. 
You must correct and modify the prompt based on the modification suggestions, if they are provided. The new prompt must be unique from all previous prompts in both the prompt history and what you come up with.
Think carefully about how you can modify the current prompt given the information available to you. 

### Prompt History ###

{prompt_history}

### Current Prompt ###

{current_prompt}

### Modification Suggestions ###

{suggestions}

###Instructions###

1. You will generate a new prompt based on the modficiation suggestions. If no suggestions are supplied, then just go to steps 3 and 4.
2. Follow the analysis suggestions exactly and a predicted score for this prompt.
3. The new prompt must be different from all of the previous prompts.
4. The new prompt must be modified to prevent the failure cases.

You must follow the evaluation instructions exactly! Do not deviate from the suggestions, even if they seem opposite to what
you would do. Your task is just to implement the suggestions not come up with your own solution at this step.

PromptVariants:

"""

# TODO: Probably put these elsewhere but for now keep here


def build_optimizer_prompt(
    task_desc,
    optimize_job,
    output_schema={
        "suggestions": "No suggestions at this time. Just create a variant."
    },
):
    return OPTIMIZER_TEMPLATE.format(
        task_desc=json.dumps(task_desc, indent=2),
        optimize_job=json.dumps(optimize_job, indent=2),
        suggestions=json.dumps(output_schema, indent=2),
    )


def build_generator_prompt(
    task_desc,
    generator_job,
    output_schema,
    optional_example="No Optional Examples for now",
    n=10,
):
    return OPTIMIZER_DATA_GENERATOR.format(
        n=n,
        generator_job=json.dumps(task_desc, indent=2),
        task_description=json.dumps(generator_job, indent=2),
        optional_examples=optional_example,
        output_schema=json.dumps(output_schema, indent=2),
    )


def build_evaluation_prompt(
    prompt, task_description, metric_results, batch_predictions, ground_truth
):
    return EVALUATE_INSTRUCTIONS.format(
        n=len(batch_predictions.predictions),
        prompt=prompt,
        task_description=task_description,
        predictions=batch_predictions,
        metric_results=metric_results,
        ground_truth=ground_truth,
    )


def build_classifier_prompt(prompt, batch):
    return CLASSIFICATION_INSTRUCTIONS.format(prompt=prompt, batch=batch)


def build_modification_prompt(prompt_hist, curr_prompt, suggestions, n=3):
    return PROMPT_MODIFICATION.format(
        n=n,
        prompt_history=prompt_hist,
        current_prompt=curr_prompt,
        suggestions=suggestions,
    )
