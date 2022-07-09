from fastapi import APIRouter

from src.api.schema import (
    InputText,
    TaskStatus,
    CalcSimilarityInput,
    CalcSimilarityResponse,
)
from src.worker.worker import vectorize_text
from src.ml.metrics import SimilarityCalculator


router = APIRouter(
    prefix='/text-vectorizer',
    tags=['Text Vectorizer']
)


@router.post(
    '/similarity',
    response_model=CalcSimilarityResponse,
    response_model_exclude_unset=True
)
def calc_text_similality(text_input: CalcSimilarityInput):
    return CalcSimilarityResponse(
        text_similarity=SimilarityCalculator.calc_similarity(
            text_input.sentence1, text_input.sentence2
        )
    )


@router.post(
    '/vectorize',
    response_model=TaskStatus,
    response_model_exclude_unset=True
)
def vectorize(text_input: InputText):
    task = vectorize_text.delay(text_input.sentence)
    return TaskStatus(id=task.id)


@router.get('/vectorize/{task_id}', response_model=TaskStatus)
def check_status(task_id: str):
    result = vectorize_text.AsyncResult(task_id)
    return TaskStatus(
        id=task_id,
        status=result.status,
        result=result.result
    )
