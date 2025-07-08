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
        description="Instructions to give to the agent on what type of task this type of prompt is in association with. for example: this prompt is about movie ranking and classification."
    )


class PromptOptimizationRequest(OptimizationRequest):
    prompt: str = Field(description="Prompt to be optimized by the agent")
    examples: Optional[dict] = Field(
        default=None,
        description="Optional list of examples to better tune the optimization to specific tasks.",
    )


class OptimzationModelMap(BaseModel):
    # idea of some storage to keep prompt context around which could be brought back up for the LLM to use.
    params: dict = Field(
        description="Dictionary of params to help a model stay tuned to the task. IE. prompt plus any additional domain information."
    )
    

class DataPoint(BaseModel):
    example: str = Field(description="Stores the example that was generated for the dataset.")
    label: str = Field(description="Label for the example.")

class DataGenerationOutput(BaseModel):
    generated_dataset: list[DataPoint] = Field(description="Field to list the generated dataset examples based on the task description.")    
