# Open RAG Assistant Backend
This is the backend service of a conversational assistant designed to improve access to public information using 
Retrieval-Augmented Generation (RAG) [Frontend](https://github.com/Stevencibambo/rag-assistant-frontend). 
It uses FastAPI for the API layer 
and Ollama to run the Mistral LLM locally. The system allows loading and querying 
documents to answer questions in natural language.

## 🚀 Features
- Load and embed multiple documents (PDF, DOCX, TEXT, MD, etc.)
- Local LLM using Ollama + Mistral or Llama
- Vector similarity search using ChromaDB
- FastAPI-based REST API
- Ready for integration with any frontend

## ⚙️ Requirements
- Python 3.10+
- Ollama installed and running
- mistral model pulled in Ollama:

> ollama pull mistral

or
> ollama run mistral

if you want uses llama changes mistral to llama3.2

## Installation
Clone the repository
> git clone https://github.com/Stevencibambo/open-rag-assistant.git

> cd open-rag-assistant

Create virtual environment
> conda create -n env_name python=3.10

> conda activate env_name

Install dependencies
> pip install -r requirements.txt

## Run Ollama (if not already running)
> ollama serve

Before start the API server remember to initialized the vector db
> python reset_chroma.py

then transform the documents (consider putting all the documents in docs directory)
> python transformer.py

## Start the API server
> uvicorn app.main:app --reload

the API will be available at: http://localhost:8000/docs


