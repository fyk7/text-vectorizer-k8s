from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union

class CalcSimilarityInput(BaseModel):
    sentence1: List[Union[float, int]]
    sentence2: List[Union[float, int]]

class CalcSimilarityResponse(BaseModel):
    text_similarity: Union[float, int]

class InputText(BaseModel):
    sentence: str

class OutputVector(BaseModel):
    vectorized_text: float = Field(..., min_items=256, max_items=256)

class TaskStatus(BaseModel):
    id: str
    status: Optional[str]
    result: Optional[Any]
