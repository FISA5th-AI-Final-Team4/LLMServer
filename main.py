from fastapi import FastAPI

from api.router import api_router


app = FastAPI()
app.include_router(api_router, prefix="/llm")


@app.get("/")
def root() -> dict[str, str]:
    return {'message': "Woori FISA 5th - Team 4 LLM Query Routing Server"}
