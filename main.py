from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from langchain_community.retrievers import YouRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

yr = YouRetriever()
model = "gpt-3.5-turbo-16k"
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=model, max_tokens=100), chain_type="stuff", retriever=yr
)

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def read_item(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail={"msg": "Message is required"})
    try:
        response = qa.run(f"You are an assistant at GVSU. {request.message}")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
