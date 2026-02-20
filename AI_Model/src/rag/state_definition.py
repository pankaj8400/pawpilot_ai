from langchain_community.vectorstores import FAISS
class RAGState:
    """
    State object for RAG Pipeline
    """

    # Input
    query: str                             # User's original question
    embedding_model: str              # Embedding model instance

    # RAG Outputs
    retrieved_docs: list                   # Retrieved documents
    formatted_context: str                 # Formatted context for prompt

    #Retriever
    db: FAISS                              # Data Base for Chroma DB
    docs: list                             # Documents 
