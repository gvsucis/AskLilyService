import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic import BaseModel
import logging

load_dotenv()

logger = logging.getLogger(__name__)

search = GoogleSearchAPIWrapper()
model = "gpt-4o-mini"
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.post("/chat")
async def query_openai(request: ChatRequest):
    message_history = RedisChatMessageHistory(
        request.session_id, url=os.getenv("REDISCLOUD_URL")
    )
    memory = ConversationBufferWindowMemory(
        memory_key="history",
        chat_memory=message_history,
        k=3,  # Remember last 3 interactions
    )
    conversation_chain = ConversationChain(
        llm=ChatOpenAI(model=model, max_tokens=100),
        memory=memory,
    )
    retrieved_info = search.results(request.message, 5)

    prompt_template = (
        "You are a helpful assistant. Using the following information: {retrieved_info}\n"
        "Answer the user's question: {user_query}"
    )

    full_prompt = prompt_template.format(
        retrieved_info=(
            retrieved_info if retrieved_info else "No relevant information found."
        ),
        user_query=request.message,
    )

    with get_openai_callback() as cb:
        response = conversation_chain(full_prompt)

    logger.info(
        (
            f"Tokens Used: {cb.total_tokens}\n"
            f"\tPrompt Tokens: {cb.prompt_tokens}\n"
            f"\tCompletion Tokens: {cb.completion_tokens}\n"
            f"Successful Requests: {cb.successful_requests}\n"
            f"Total Cost (USD): ${cb.total_cost}"
        )
    )
    return {"response": response["response"]}
