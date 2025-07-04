from typing import Optional
from pydantic import BaseModel

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
    jury_decisions: list[JuryDecision] # judge takes in decisions from jury, evidence from defense and prosecutor
    prosecutor_arguments: list[ProsecutorArgument]
    defense_arguments: list[DefenseArgument]
    decisions: None
    explanation: str
 
    
class OptimizationResult(BaseModel):
    optimized_prompt: str
    optimization_history: list[dict] # store the list of optimization history
    original_prompt: str
    reason: str
    stop: bool
    

    
class OptimizationRequest(BaseModel):
    text: str
    examples: Optional[dict]
    
    