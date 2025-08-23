import mlflow
import json
from dotenv import load_dotenv
from jurymind.core.optimization import PromptOptimizationPolicy
from jurymind.core.models import TaskExample


if __name__ == "__main__":
    load_dotenv()

    with open("small_data.json", "r") as f:
        dataset = json.load(f)

    task_examples = []  # todo build factory method for this
    for elm in dataset:
        task_examples.append(TaskExample(example=elm["review"], label=elm["label"]))

    policy = PromptOptimizationPolicy(
        "Classify the following data to see if they contain spoilers or not. Label should be 0 or 1.",
        "The task is a binary classification task to check if a review has spoilers in them or not.",
        evaluation_examples=task_examples,
    )

    policy.run()

    print(policy.get_step_history())
    print()
    print(policy.get_optimized_prompt())
