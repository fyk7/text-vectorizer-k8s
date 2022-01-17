from fastapi import APIRouter
from src.api.schema import (
    InputText,
    TaskStatus,
    CalcSimilarityInput,
    CalcSimilarityResponse,
)
from src.worker.worker import vectorize_text
from src.worker.worker import create_task
from src.libs.vectorize_text import TextVectorizer
from fastapi.responses import JSONResponse
from fastapi import Body


router = APIRouter(
    prefix='/vectorize-text',
    tags=['vectorize-text']
)

# Light weight task
@router.post(
    '/text-similarity',
    response_model=TaskStatus,
    response_model_exclude_unset=True
)
def calc_text_similality(text_input: CalcSimilarityInput):
    text_similarity = \
        TextVectorizer.calc_similarity(text_input.sentence1, text_input.sentence2)
    return CalcSimilarityResponse(text_similarity)

# Heavy weight task
@router.post(
    '/vectorize',
    response_model=TaskStatus,
    response_model_exclude_unset=True
)
def vectorize(text_input: InputText):
    task = vectorize_text.delay(text_input.sentence)
    return TaskStatus(id=task.id)


@router.get('/{task_id}', response_model=TaskStatus)
def check_status(task_id: str):
    result = vectorize_text.AsyncResult(task_id)
    status = TaskStatus(
        id=task_id,
        status=result.status,
        result=result.result
    )
    return status


@router.post("/test-tasks", status_code=201)
def run_task(payload = Body(...)):
    task_type = payload["type"]
    task = create_task.delay(int(task_type))
    return JSONResponse({"task_id": task.id})


@router.get("/test-tasks/{task_id}")
def get_status(task_id: str):
    task_result = create_task.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)