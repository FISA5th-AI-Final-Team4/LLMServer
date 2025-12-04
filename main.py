from fastapi import FastAPI

from core.setup import lifespan
from api.router import api_router


app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/llm")


@app.get("/")
def root() -> dict[str, str]:
    return {'message': "Woori FISA 5th - Team 4 LLM Query Routing Server"}
