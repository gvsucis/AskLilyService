# Ask Lily API - Laker Mobile

This is a basic HTTP API that allows users to chat with an OpenAI LLM that provides answers to prompts that are enriched with You.com's Search API. The answers are intended to be related to GVSU.

## Project setup

1. After cloning the repository, create a python virtual environment in the root directory: `python -m venv .venv`
2. Download the dependencies: `pip install -r requirements.txt`.
3. Create a `.env` file from `.env.template` and update it with your environment variables.

## Running the project

To run the project, just execute `fastapi dev main.py`. Then, you can test the api with a request like:
```
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "What is the ACI?"}'
```
