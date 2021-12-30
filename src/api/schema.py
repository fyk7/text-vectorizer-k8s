from pydantic import BaseModel
from typing import List, Optional, Any

class InputText(BaseModel):
    sentence: str

class OutputVector(BaseModel):
    # TODO 可能なら返却するベクトルの長さを指定する
    vectorized_text: List[float]

class TaskStatus(BaseModel):
    id: str
    status: Optional[str]
    result: Optional[Any]
