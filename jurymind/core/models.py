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
    decision: str
    explanation: str
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
    original_prompt: str = Field(
        description="Orginal prompt that the agent was given to optimize."
    )
    reason: str = Field(
        description="Detailed explanation for the changes and why the changes were needed."
    )
    confidence_score: str = Field(
        description="Score based on the Likert scale between 1 to 5 on how confident you are in the change being better than previous prompt."
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


class PromptOptimizationRequest(OptimizationRequest):
    prompt: str = Field(description="Prompt to be optimized by the agent")
    task_description: str = Field(description="An explanation of the task the promt is attemtping to perform.")
    iterations: int = Field(description="The number of iterations to perform the optimization on.")

class OptimzationModelMap(BaseModel):
    # idea of some storage to keep prompt context around which could be brought back up for the LLM to use.
    params: dict = Field(
        description="Dictionary of params to help a model stay tuned to the task. IE. prompt plus any additional domain information."
    )


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
        description="You put the labels here for the generated examples."
    )


class SampleAnalysis(BaseModel):
    analysis: str
    reasoning: str = Field(
        description="A detailed and concise 2-3 sentence explanation of why you came to this analysis."
    )
    prediction: int = Field(description="Boolean prediction of a sample of data.")


class ClassificationResult(BaseModel):
    explanation: str = Field(description="Your explanation for why you made your prediction")
    sample: str = Field(
        description="The sample that is to be classified according to the task."
    )
    prompt: str = Field(description="Prompt used to classify the sample.")
    prediction: int = Field(
        description="You come up with a binary prediction of 0 or and 1 for this sample. This is not where you put the ground truth."
    )
    # ground_truth: int = Field(description="The actual ground truth classification label for the example.")


class OptimizationStep(BaseModel):
    pass


class OptimizationStepResult(BaseModel):
    modified_prompt: str = Field(description="The modified prompt you came up with to improve the prompt.")
    explanation_of_changes: str = Field(description="You must give a reason for the changes you made and why it will work better.")



class BatchClassificationResult(BaseModel):
    predictions: list[ClassificationResult]
    


class ClassificationReport(BaseModel):
    prompt: str = Field(
        description="The prompt that was used for the task on the examples."
    )
    
    suggested_changes: str = Field(
        description="Changes that should be made to the original prompt to improve its ability to perform the task. Should be itemized and given a good explanation for the suggestions."
    )
    accuracy: float = Field(
        description="The accuracy percentage of the classification results to the true label between 0 and 1."
    )
    confusion_matrix: dict = Field(
        description="Confusion matrix of the predictions to the ground truth."
    )
    
    incorrect: list[ClassificationResult] = Field(description="You put the examples that were incorrectly classified by the llm as a list of ClassificationResult objects.")
