import os
import cohere
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.retrievers import YouRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel
import requests

load_dotenv()

yr = YouRetriever()
model = "gpt-3.5-turbo-16k"
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=model, max_tokens=100), chain_type="stuff", retriever=yr
)

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def query_openai(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail={"msg": "Message is required"})
    try:
        response = qa.run(f"You are an assistant at GVSU. {request.message}")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_ai_snippets_for_query(query):
    headers = {"X-API-Key": os.environ["YDC_API_KEY"]}
    results = requests.get(
        f"https://api.ydc-index.io/search?query={query}",
        headers=headers,
    ).json()

    # We return many text snippets for each search hit so
    # we need to explode both levels
    return "\n".join(["\n".join(hit["snippets"]) for hit in results["hits"]])


def get_cohere_prompt(query, context):
    return f"""You are an assistant at GVSU. Given a question and a bunch of snippets context try to answer the question using the context. If you can't please say 'Sorry hooman, no dice'.
question: {query}
context: {context}
answer: """


def ask_cohere(query, context):
    try:
        response = cohere_client.generate(prompt=get_cohere_prompt(query, context))
        if response.generations:
            return response.generations[0].text
        else:
            return "Sorry hooman, no dice"
    except Exception as e:
        print("Cohere call failed for query {} and context {}".format(query, context))
        print(e)
        return "Sorry hooman, no dice"


def ask_cohere_with_ai_snippets(query):
    ai_snippets = get_ai_snippets_for_query(query)
    return ask_cohere(query, ai_snippets)


@app.post("/chat_co")
def query_cohere(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail={"msg": "Message is required"})
    try:
        response = ask_cohere_with_ai_snippets(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
