from pathlib import Path
from typing import Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import schemas, settings


def _load_doc(file_path: Path) -> list[Document]:
    """Load PDF document for embedding extraction.

    Args:
        file_path (Path): Path to document.

    Returns:
        list[Document]: List of loaded text data from PDF file.
    """
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()


def _preprocess_text(
    documents: list[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[Document]:
    """Preprocess loaded text data and split to chunks.

    Args:
        documents (list[Document]): Loaded text data.
        chunk_size (int, optional): Chunk size. Defaults to settings.CHUNK_SIZE.
        chunk_overlap (int, optional): Chunk overlap. Defaults to settings.CHUNK_OVERLAP.

    Returns:
        list[Document]: List of preprocessed text data split to chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents=documents)


def _get_embeddings(
    documents: list[Document],
    batch_size: int = settings.BATCH_SIZE,
    model: str = settings.EMBEDDING_MODEL,
) -> VectorStore:
    """Extract embeddings from text data.

    Args:
        documents (list[Document]): Preprocessed text data.
        batch_size (int, optional): Batch size. Defaults to settings.BATCH_SIZE.
        model (str, optional): Embeddings Model. Defaults to settings.EMBEDDING_MODEL.

    Returns:
        VectorStore: Extracted embeddings in vectore store.
    """
    embedding = OpenAIEmbeddings(model=model)
    vector_embeddings = None

    # Process and store embeddings in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        if vector_embeddings is None:
            vector_embeddings = FAISS.from_documents(
                documents=batch, embedding=embedding
            )
        else:
            vector_embeddings.add_documents(documents=batch)
    return vector_embeddings


def _store_embeddings(
    vector_embeddings: VectorStore, vector_db: Path = settings.VECTOR_DB
) -> None:
    """Store extracted embeddings to vector storage locally.

    Args:
        vector_embeddings (VectorStore): Vector embeddings.
        vector_db (Path, optional): Local storage. Defaults to settings.VECTOR_DB.
    """
    vector_embeddings.save_local(vector_db)


def upload_pdf_doc(file_path: Path) -> None:
    """Upload -> extract -> store PDF to vector storage pipeline.

    Args:
        file_path (Path): Path to PDF document.
    """
    documents = _load_doc(file_path=file_path)
    chunks = _preprocess_text(documents=documents)
    embeddings = _get_embeddings(documents=chunks)
    _store_embeddings(vector_embeddings=embeddings)


def load_embeddings(
    vector_db: Path = settings.VECTOR_DB, model: str = settings.EMBEDDING_MODEL
) -> VectorStore:
    """Load embedding from local vector storage.

    Args:
        vector_db (Path, optional): Vector storage. Defaults to settings.VECTOR_DB.
        model (str, optional): Embeddings model. Defaults to settings.EMBEDDING_MODEL.

    Returns:
        VectorStore: Loaded embeddings.
    """
    embedding = OpenAIEmbeddings(model=model)
    return FAISS.load_local(
        vector_db, embeddings=embedding, allow_dangerous_deserialization=True
    )


def similarity_search(
    query: str, vector_embeddings: VectorStore, k: int = 4
) -> list[Document]:
    """Similarity search for input text and vector store.

    Args:
        query (str): Input text.
        vector_embeddings (VectorStore): Vector store.
        k (int, optional): Number of top closest embeddings. Defaults to 4.

    Returns:
        list[Document]: List of top similar text chunks.
    """
    return vector_embeddings.similarity_search(query=query, k=k)


def _get_model() -> BaseChatOpenAI:
    """Instantiate LLM model for conversation.

    Returns:
        BaseChatOpenAI: Chat model instance.
    """
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    )


def _create_rag_chain(system_prompt: str = settings.SYSTEM_PROMPT) -> Any:
    """Create conversation chain with LLM model, embeddings and predefined prompt.

    Args:
        system_prompt (str, optional): Prompt. Defaults to settings.SYSTEM_PROMPT.

    Returns:
        Any: RAG chain.
    """
    model = _get_model()
    retriever = load_embeddings().as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def prompt_query(query: schemas.InputPrompt) -> schemas.Answer:
    """Chat function with RAG chain and input prompt.

    Args:
        query (schemas.InputPrompt): Input prompt.

    Returns:
        schemas.Answer: LLM answer.
    """
    chain = _create_rag_chain()
    return chain.invoke(query)
