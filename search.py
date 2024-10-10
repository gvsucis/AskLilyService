import os

import boto3
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import BedrockChat
from langchain_google_community import GoogleSearchAPIWrapper


load_dotenv()

search = GoogleSearchAPIWrapper()

redis_host = os.getenv("REDIS_HOST")
redis_port = int(os.getenv("REDIS_PORT"))
redis_password = os.getenv("REDIS_PASSWORD")

session_id = "user_11"

message_history = RedisChatMessageHistory(
    session_id, url=f"redis://:{redis_password}@{redis_host}:{redis_port}"
)

memory = ConversationBufferWindowMemory(
    memory_key="history",
    chat_memory=message_history,
    k=3,  # Remember last 3 interactions
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
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

conversation_chain = ConversationChain(
    llm=bedrock_model,
    memory=memory,
)


def get_response_with_retrieval(query):
    retrieved_info = search.results("Jonathan Engelsma", 5)

    prompt_template = (
        "You are a helpful assistant at GVSU. Using the following information: {retrieved_info}\n"
        "Answer the user's question: {user_query}"
    )

    full_prompt = prompt_template.format(
        retrieved_info=(
            retrieved_info if retrieved_info else "No relevant information found."
        ),
        user_query=query,
    )

    return conversation_chain(full_prompt)


response = get_response_with_retrieval("Who is Jonathan Engelsma?")
print(response["response"])

response = get_response_with_retrieval("What does he teaches?")
print(response["response"])
