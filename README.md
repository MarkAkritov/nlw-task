# Building RAG System for Question Answering on PDF

- [Building RAG System for Question Answering on PDF](#building-rag-system-for-question-answering-on-pdf)
  - [Overview](#overview)
  - [Architecture](#architecture)
    - [RAG Pipeline](#rag-pipeline)
    - [Backend](#backend)
    - [Frontend](#frontend)
  - [Dependencies and requirements](#dependencies-and-requirements)
    - [Running API and UI](#running-api-and-ui)
  - [Potential Enhancements](#potential-enhancements)

**Objective:**

> Develop a Retrieval-Augmented Generation (RAG) system that allows users to upload a PDF document, stores its content in a vector database, and enables them to ask questions related to the document.

## Overview

This project implements a pipeline that processes uploaded PDF documents, extracts their textual content, generates embeddings and stores them to a local vector store  for question answering.

## Architecture

Repository contains API and minimal UI implementations, with `FastAPI` for backend services and  `Streamlit` for a UI interface. The core functionality relies on `LangChain`, `FAISS`, and `OpenAI`'s LLM to deliver precise responses.
Both backend and frontend services are Dockerized/orchestrated using `docker-compose`.
Source code uses `git` as version control system and `pre-commit` as git hook provider for auto formatting and linting.

Codes in repository are organized in the following way:

```txt
├── api
│   ├── Dockerfile
│   ├── __init__.py
│   ├── main.py
│   ├── requirements.txt
│   ├── schemas.py
│   ├── settings.py
│   └── utils.py
├── app
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── docker-compose.yaml
├── Pipfile
├── Pipfile.lock
├── README.md
├── embeddings.ipynb
└── requirements.txt
```

`api` directory contains codes for RAG pipeline (`utils.py`) and API implementation (`api/main.py`).
Configuration is done using `settings.py`. Specifically:

- `EMBEDDING_MODEL`: Name of the model to be used for embeddings
- `VECTOR_DB`: Path to Vector DB storage
- `DOCS_DIR`: Path to store PDF documents
- `CHUNK_SIZE`: Chunk size to split documents' text
- `CHUNK_OVERLAP`: Chunk overlap between splits
- `BATCH_SIZE`: Batch size for sending chunks to embedding model
- `SYSTEM_PROMPT`: Initial prompt for LLM

Secrets and environment variables, such as OpenAI API keys are kept in `api/.env` file and loaded in `settings.py`. `api/.env.template` file serves as a template for creating local `.env`.

### RAG Pipeline

- `LangChain` has been used for document splitting, embedding extraction, and LLM responses.
- `FAISS` was used as a vector database for storing, retrieving embeddings and similarity search.
- `OpenAI`'s `text-embedding-ada-002` model was used for embeddings.
- `OpenAI`'s `GPT 4o` LLM was provided for user queries and contextual responses.

### Backend

API handles PDF parsing, text loading, processing, chunking, and embedding generation.
Executes similarity search and answers user queries using an LLM.

- `FastAPI`: as a framework. API contains two `POST` requests:
    1. `api/upload-pdf`: request for uploading PDF document to Vector DB.
    2. `api/prompt`: request answers for provided queries (arg. `{input: question}`).
   Detailed docs can be found in `localhost:8000/docs` when API is running.

### Frontend

UI web app was built using `Streamlit`. Users can upload PDF documents and enter questions related to those documents. UI includes PDF uploading section, conversational history and responses in an interactive chat section.

## Dependencies and requirements

- Programming language: Python (3.12)
- Libraries and tools are listed in `Pipfile` and freezed in `requirements.txt`. to install, run `pip install -r requirements.txt` preferrably in a virtual environment
- `.env` file in `./api/` directory following `./.env.template` for setting up envorinment variables

### Running API and UI

Running locally from root directory:

```bash
# API: localhost:8000
> uvicorn api.main:app
# Web app: localhost:8501
> streamlit run app/main.py
```

Running using Docker, Docker Compose:

```bash
> docker-compose up -d --build
```

> Note: When running using Docker, `API_URL` variable in `./app/main.py` should be changed to "http://api:8000"`.

## Potential Enhancements

While RAG pipeline key features include document embedding retrieval, contextual answers and user interaction, project's implementation is minimal and there is a number of potential improvements and anhancement listed below.

1. Add support for processing multiple PDFs in one session, appending embeddings to existing vector store.

2. System prompt engineering for better assisting of the LLM, default answers etc.

3. Better embedding retrieval implementation, for fast and efficient similarity search and storage.

4. Memory aware chat conversation.

5. Secure application by adding authentication and password access.

**Conclusion:**

This project showcases minimal implementation for a practical application of LLM based intelligent, PDF document question answering and assisting system. Modular approach, combining with a API backend and an interactive frontend, enables usability, testing, and user queries.
