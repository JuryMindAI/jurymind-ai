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

# schema as raw string (no need to format it)
formatted_schema = json.dumps(OptimizationStepResult.model_json_schema(), indent=2)

# only parameterize this part
OPTIMIZER_TEMPLATE = """Your job is to optimize a prompt from a user. You optimize 
by seeing how to rewrite, fix, or enhance the prompt to best work with an LLM and perform the required task.

Below is the request to optimize format with field descriptions:

{task_desc}

Request to optimize values:

{optimize_job}

Output your results like so:
"""

# build the final string
OPTIMIZER_TEMPLATE = OPTIMIZER_TEMPLATE.format(
    task_desc="hello",
    optimize_job="HHH"
) + f"\n{formatted_schema}\n\nresult:"

import json

def __build_optimizer_prompt(task_desc, optimize_job, schema):
    return OPTIMIZER_PROMPT_TEMPLATE.format(
        task_desc=json.dumps(task_desc, indent=2),
        optimize_job=json.dumps(optimize_job, indent=2),
        schema=json.dumps(schema, indent=2)
    )
