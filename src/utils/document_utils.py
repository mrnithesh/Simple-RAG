import os
import shutil
from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.constants import CHROMA_DIR, FIXED_PROJECT_NAME, FIXED_PDF_PATH


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    # Cached to avoid reloading the embedding model on every run
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _get_chroma(project: str) -> Chroma:
    project_dir = os.path.join(CHROMA_DIR, project)
    return Chroma(
        persist_directory=project_dir,
        embedding_function=_get_embeddings(),
        collection_name=project,
    )


def _get_chroma_light(project: str) -> Chroma:
    # Return a Chroma instance without loading embeddings (for quick metadata checks).
    project_dir = os.path.join(CHROMA_DIR, project)
    return Chroma(
        persist_directory=project_dir,
        collection_name=project,
    )


def initialize_fixed_document():
    # Initialize vector store
    if not os.path.exists(FIXED_PDF_PATH):
        return False, f"PDF file not found at {FIXED_PDF_PATH}. Please place your PDF file there."

    project_dir = os.path.join(CHROMA_DIR, FIXED_PROJECT_NAME)

    # Fast path: if collection exists with vectors, skip processing
    try:
        light_vs = _get_chroma_light(FIXED_PROJECT_NAME)
        count = getattr(light_vs, "_collection", None)
        if count is not None and light_vs._collection.count() > 0:
            return True, "Document already loaded and processed."
    except Exception:
        # If any error, fall back to processing logic below
        pass

    try:
        docs = load_and_chunk(FIXED_PDF_PATH)
        vectordb = _get_chroma(FIXED_PROJECT_NAME)
        vectordb.add_documents(docs)
        # Ensure persistence if supported
        try:
            persist = getattr(vectordb, "persist", None)
            if callable(persist):
                persist()
        except Exception:
            pass
        return True, f"Successfully processed {len(docs)} document chunks from {FIXED_PDF_PATH}"
    except Exception as e:
        return False, f"Error processing document: {str(e)}"


def get_fixed_vectorstore() -> Chroma:
    # Return the vector store for the fixed document.
    return _get_chroma(FIXED_PROJECT_NAME)


def get_fixed_vector_count() -> int:
    # Return number of vectors stored for the fixed project, 0 if unavailable.
    try:
        vs = _get_chroma_light(FIXED_PROJECT_NAME)
        # Prefer fast internal count if present
        if getattr(vs, "_collection", None) is not None:
            return int(vs._collection.count())
        data = vs.get(include=[])
        ids = data.get("ids") if isinstance(data, dict) else []
        return len(ids) if ids else 0
    except Exception:
        return 0


def load_and_chunk(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(docs)


def clear_fixed_vectorstore() -> tuple[bool, str]:
    # Delete the persisted Chroma directory for the fixed project to force re-indexing.
    project_dir = os.path.join(CHROMA_DIR, FIXED_PROJECT_NAME)
    try:
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            return True, "Cleared existing vector store."
        return True, "No existing vector store to clear."
    except Exception as e:
        return False, f"Failed to clear vector store: {e}"