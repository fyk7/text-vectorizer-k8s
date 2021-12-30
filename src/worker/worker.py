import os
from celery import Celery
# from src.libs.vectrize_text import TextVectorizer


celery = Celery(__name__)
REDIS_URL = "redis://{host}:{port}/1".format(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=os.getenv('REDIS_PORT', '6379')
)
celery.conf.broker_url = REDIS_URL
celery.conf.result_backend = REDIS_URL


# 新しいタスクを追加したらdocker-compose up --buildすること
# @celery.task(name='tasks.vectorize_text')
# def vectorize_text(text: str) -> float:
#     text = text[:256] if len(text) > 256 else text
#     return TextVectorizer.vectorize(text).tolist()


@celery.task(name="create_task")
def create_task(task_type):
    import time
    time.sleep(int(task_type) * 5)
    return True
