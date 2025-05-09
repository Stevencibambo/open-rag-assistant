# ./app.py
# run : uvicorn main:app --reload to start the server
# access via : 127.0.0.1:8000/docs

from fastapi import FastAPI
from pydantic import BaseModel
from agent import ask_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: AskRequest):
    response = ask_agent(request.question)
    return {"response": response}
