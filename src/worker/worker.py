import os
import typing as t

import numpy as np
from celery import Celery, states
from celery.exceptions import Ignore

from src.libs.text_vectorizer import TextVectorizer


celery = Celery(__name__)
REDIS_URL = "redis://{host}:{port}/0".format(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=os.getenv('REDIS_PORT', '6379')
)
celery.conf.broker_url = REDIS_URL
celery.conf.result_backend = REDIS_URL


@celery.task(bind=True, name='tasks.vectorize_text')
def vectorize_text(self, text: str) -> t.List[float]:
    text = text[:256] if len(text) >= 256 else text
    try:
        res = TextVectorizer.vectorize(text)
        if isinstance(res, np.ndarray):
            res = res.tolist()
        return res
    except Exception as e:
        self.update_state(
            state = states.FAILURE,
            meta = e
        )
        raise Ignore()
    

@celery.task(name="create_task")
def create_task(task_type):
    import time
    time.sleep(int(task_type) * 5)
    return True
