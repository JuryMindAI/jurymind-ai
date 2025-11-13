from typing import Optional
from pydantic import BaseModel, Field


class ProsecutorArgument(BaseModel):
    evidence: list
    argument: str
    confidence: float


class DefenseArgument(BaseModel):
    evidence: list
    argument: str
    confidence: float


class JuryDecision(BaseModel):
    explanation: str
    decision: str
    confidence: float
    decision: str
    confidence: float


class JudgeDecision(BaseModel):
    jury_decisions: list[
        JuryDecision
    ]  # judge takes in decisions from jury, evidence from defense and prosecutor
    prosecutor_arguments: list[ProsecutorArgument]
    defense_arguments: list[DefenseArgument]
    decisions: None
    explanation: str


class OptimizationStepResult(BaseModel):
    optimized_prompt: str = Field(
        description="Field to store the optimized prompt the agent rewrote."
    )
    original_prompt: str = Field(description="Orginal prompt that was to be optimize.")
    reason: str = Field(
        description="Detailed explanation for the changes and why the changes were needed."
    )
    confidence_score: str = Field(
        description="Likert scale between 1 to 5 on how confident you are in the change being better than the previous prompt."
    )
    stop: bool


class OptimizationRunResult(BaseModel):
    steps: list[OptimizationStepResult]
    model_id: str


class PromptOptimizationRunResult(OptimizationRunResult):
    optimal: str
    original: str
    explanation: str
    confidence_score: str


class OptimizationRequest(BaseModel):
    task_description: str = Field(
        description="Instructions to give to the agent on what type of task the prompt is associated with."
    )


class PromptOptimizationConfig(OptimizationRequest):
    prompt: str = Field(description="Prompt that you are to optimize.")
    task_description: str = Field(
        description="An explanation of the task the prompt is attemtping to perform."
    )


class OptimzationModelMap(BaseModel):
    # idea of some storage to keep prompt context around which could be brought back up for the LLM to use.
    params: dict = Field(
        description="Dictionary of params to help a model stay tuned to the task. IE. prompt plus any additional domain information."
    )


class TaskExample(BaseModel):
    example: str = Field(description="Example to use for the Task.")
    label: str = Field(description="Label of the example for the given task.")


class DataPoint(BaseModel):
    example: str = Field(
        description="Stores the example that was generated for the dataset."
    )
    label: int = Field(
        description="Binary label for the example with 1 being true and 0 being false."
    )


class DataGenerationOutput(BaseModel):
    examples: list[str] = Field(
        description="You list the generated examples here and DO NOT inlcude the label."
    )
    labels: list[int] = Field(
        description="You put the labels here for the examples based on your prediction."
    )


class SampleAnalysis(BaseModel):
    reasoning: str = Field(
        description="A detailed and concise 2-3 sentence explanation of why you came to this analysis."
    )
    prediction: int = Field(description="Boolean prediction of a sample of data.")


class ClassificationResult(BaseModel):
    explanation: str = Field(description="Explain why you predicted the given label.")
    prediction: str = Field(
        description="You come up with a classification prediction based on the prompt instructions. This is not where you put the ground truth or other explanation."
    )


class OptimizationStepResult(BaseModel):

    explanation: str = Field(
        description="You explains the reasons for the changes you made along with how it will solve for issues with the original prompt."
    )
    modified_prompt: str = Field(
        description="The modified version you came up with to improve the original prompt."
    )
    confidence: str = Field(
        description="Your confidence level between 1 and 5 on a Likert scale that the new prompt will perform better than the previous prompt."
    )


class BatchClassificationResult(BaseModel):
    predictions: list[ClassificationResult]


class ModificationReport(BaseModel):

    suggested_changes: str = Field(
        description="Changes that should be made to the original prompt to improve its ability to perform the task. Each suggested change should be defined via a markdown list that another LLM can follow."
    )

    explanation: str = Field(
        description="You must give your reasoning as to why these changes need to be made to increase the performance on the task."
    )

    # accuracy: float = Field(
    #     description="The accuracy percentage of the classification results."
    # )

    # confusion_matrix: dict = Field(
    #     description="Confusion matrix of the predictions to the ground truth."
    # )


class PromptVariants:
    variants: list[str] = Field(description="Put the list of new variant prompts here.")
