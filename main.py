import os

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import BedrockChat
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic import BaseModel

load_dotenv()

search = GoogleSearchAPIWrapper()

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="us-east-1",
)

bedrock_model = BedrockChat(
    model_id="amazon.titan-text-lite-v1",
    client=bedrock_runtime,
    model_kwargs={
        "maxTokenCount": 256,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    },
)

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.post("/chat")
def query_openai(request: ChatRequest):
    message_history = RedisChatMessageHistory(
        request.session_id, url=os.getenv("REDISCLOUD_URL")
    )
    memory = ConversationBufferWindowMemory(
        memory_key="history",
        chat_memory=message_history,
        k=3,  # Remember last 3 interactions
    )
    conversation_chain = ConversationChain(
        llm=bedrock_model,
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

    response = conversation_chain(full_prompt)
    return {"response": response["response"]}
