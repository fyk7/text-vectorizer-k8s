import os
import importlib
import typing as t

import numpy as np
from celery import Celery, states
from celery.exceptions import Ignore


celery = Celery(__name__)
REDIS_URL = "redis://{host}:{port}/0".format(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=os.getenv('REDIS_PORT', '6379')
)
celery.conf.broker_url = REDIS_URL
celery.conf.result_backend = REDIS_URL


@celery.task(bind=True, name='tasks.vectorize_text')
def vectorize_text(self, text: str) -> t.List[float]:
    # Lazy import!
    # If TextVectorizer is imported globally,
    # you shuold install large dependencies (like torch) to FastAPI container. 
    text_vectorizer = importlib.import_module('src.ml.text_vectorizer')

    text = text[:256] if len(text) >= 256 else text
    try:
        res = text_vectorizer.TextVectorizer.vectorize(text)
        if isinstance(res, np.ndarray):
            res = res.tolist()
        return res
    except Exception as e:
        self.update_state(
            state = states.FAILURE,
            meta = e
        )
        raise Ignore()
    