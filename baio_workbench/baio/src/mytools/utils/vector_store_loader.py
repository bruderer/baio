from langchain.vectorstores import FAISS


def load_vector_store(vector_store_path: str, embedding):
    """
    Load a vector store from a local path.

    Args:
    vector_store_path (str): The path to the vector store.
    embedding: The embedding model to use.

    Returns:
    FAISS: The loaded vector store.
    """
    return FAISS.load_local(vector_store_path, embedding)
