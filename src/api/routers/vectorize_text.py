from fastapi import APIRouter
from src.api.schema import InputText, TaskStatus
from celery.result import AsyncResult
# from src.worker.worker import vectorize_text
from src.worker.worker import create_task
from fastapi.responses import JSONResponse
from fastapi import Body


# TODO prefixをmain側で宣言するか以下で宣言するかを考察
router = APIRouter(
    prefix='/vectorize-text',
    tags=['vectorize-text']
)


# TODO viewの関数名やurlのパスに統一感を持たせる
# @router.post(
#     '/vectorize',
#     response_model=TaskStatus,
#     response_model_exclude_unset=True
# )
# def vectorize(text_input: InputText):
#     task = vectorize_text.delay(text_input.sentence)
#     return TaskStatus(id=task.id)

# @router.get('/{task_id}', response_model=TaskStatus)
# def check_status(task_id: str):
#     # celeryに対して特定のtaskが完了している場合結果を取得する
#     result = AsyncResult(task_id)
#     status = TaskStatus(
#         id=task_id,
#         status=result.status,
#         result=result.result
#     )
#     return status

@router.post("/tasks", status_code=201)
def run_task(payload = Body(...)):
    task_type = payload["type"]
    task = create_task.delay(int(task_type))
    # 一旦task idだけ返却して後で結果を取得する
    return JSONResponse({"task_id": task.id})


@router.get("/tasks/{task_id}")
def get_status(task_id: str):
    print("GET was called !!!!!!!!!!")
    task_result = create_task.AsyncResult(task_id)
    print(task_result)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)