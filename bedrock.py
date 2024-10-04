import os
import boto3
from dotenv import load_dotenv
from langchain_community.retrievers import YouRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

import os
from dotenv import load_dotenv
from langchain_community.retrievers import YouRetriever
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv()

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", "eYVX7EwVmmxKPCDmwMtyKVge8oLd2t81")

# Set up Redis-backed memory
session_id = "user_3"  # Generate or retrieve a unique session ID

message_history = RedisChatMessageHistory(
    session_id, url=f"redis://:{redis_password}@{redis_host}:{redis_port}"
)

memory = ConversationBufferWindowMemory(
    memory_key="history",
    chat_memory=message_history,
    k=3,  # Remember last 5 interactions
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

# Create a ConversationChain instance without the retriever
conversation_chain = ConversationChain(
    llm=bedrock_model,
    memory=memory,
)

# Initialize the retriever
yr = YouRetriever()


def get_response_with_retrieval(query):
    # Fetch relevant information using the retriever
    retrieved_info = yr.results("GVSU: " + query)

    prompt_template = (
        "You are a helpful assistant at GVSU. Using the following information: {retrieved_info}\n"
        "Answer the user's question and provide links when possible: {user_query}"
    )

    full_prompt = prompt_template.format(
        retrieved_info=(
            retrieved_info[:2] if retrieved_info else "No relevant information found."
        ),
        user_query=query,
    )

    return conversation_chain(full_prompt)


# Use the conversation chain with retrieval logic
response = get_response_with_retrieval("Who is Jonathan Engelsma?")
print(response["response"])

response = get_response_with_retrieval("What does he teaches?")
print(response["response"])
