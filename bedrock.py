import os
import boto3
from dotenv import load_dotenv
from langchain_community.retrievers import YouRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

load_dotenv()

# Redis setup
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", "eYVX7EwVmmxKPCDmwMtyKVge8oLd2t81")

# YouRetriever and Bedrock setup
yr = YouRetriever()

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

bedrock_model = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime,
    model_kwargs={
        "max_tokens": 256,
    },
)

# Set up Redis-backed memory
session_id = "user_1234"  # Generate or retrieve a unique session ID

message_history = RedisChatMessageHistory(
    session_id, url=f"redis://:{redis_password}@{redis_host}:{redis_port}"
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    k=5,  # Remember last 5 interactions
)

# Create the RetrievalQA chain with Redis-backed memory
qa = RetrievalQA.from_chain_type(
    llm=bedrock_model, chain_type="stuff", retriever=yr, memory=memory
)

# Use the chain
response = qa(
    "You are a helpful assistant at GVSU. Answer the following question from a user: What is ACI?"
)
print(response["result"])

response = qa("Now answer this question: List some of their projects")
print(response["result"])

print("------------------------------")
print(memory.buffer)
