from fastapi import FastAPI
from src.api.routers.vectorize_text import router as vectorize_router

app = FastAPI(
    title="vectorize_text_api"
)
app.include_router(vectorize_router)

@app.get("/ping", summary="Check that the service is operational")
def pong():
    return {"ping": "pong"}
