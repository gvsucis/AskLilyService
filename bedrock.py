import boto3
from dotenv import load_dotenv
from langchain_community.retrievers import YouRetriever
from langchain.chains import RetrievalQA
from langchain.memory.buffer_window import ConversationBufferWindowMemory

from langchain_community.llms import Bedrock

load_dotenv()

yr = YouRetriever()

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

bedrock_model = Bedrock(
    model_id="amazon.titan-text-lite-v1",
    client=bedrock_runtime,
    model_kwargs={
        "maxTokenCount": 512,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    },
)

memory = ConversationBufferWindowMemory(k=3)  # Remembers the last 3 interactions

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=bedrock_model, chain_type="stuff", retriever=yr, memory=memory
)

response = qa.run(
    f"You are an assistant at GVSU. When did the current academic semester start?"
)
print(response)

response = qa.run(f"And when does it end?")
print(response)

print(memory.buffer)
