import mlflow
import json
from dotenv import load_dotenv
import pandas as pd
from jurymind.core.optimization import PromptOptimizer
from jurymind.core.models import ClassificationResult, TaskExample
from jurymind.evaluation.base import evaluator
from sklearn.metrics import accuracy_score


@evaluator
def scorer(expected_result, output_result):
    """
    scorer _summary_

    Args:
        expected_result (_type_): _description_
        output_result (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("HAHAHH")


def accuracy_evaluator(output: list[ClassificationResult], expectations: list[str]):
    """Takes outputs and expectations and returns accuracy measurement"""

    correct = 0
    total = 0
    for i, x in enumerate(output):

        if x.prediction.lower() == expectations[i].lower():
            correct += 1
        total += 1
    return float(correct) / total


if __name__ == "__main__":
    load_dotenv()

    df = pd.read_csv("spamhamdata.csv", sep="\t", header=None, names=["label", "sms"])

    n_samples = 30

    sample = df.groupby("label").sample(n=n_samples, random_state=42)
    sample = sample.sample(frac=1)
    task_exmamples = [
        TaskExample(example=x.sms, label=x.label) for x in sample.itertuples()
    ]
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment("Test1")
    policy = PromptOptimizer(
        "Tell me if the following data is spam or ham.",
        "The task is a binary classification task to determine if some piece of data fits the prompts criteria.",
        evaluation_examples=task_exmamples,
        evaluators=[accuracy_evaluator],
        max_epochs=2,
    )

    policy.run()

    print(policy.get_step_history())
    print()
    print(policy.get_optimized_prompt())
    scorer("", "")
