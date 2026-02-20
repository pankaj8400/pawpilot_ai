import os
import time
from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from embeddings import embed_query
from dotenv import load_dotenv
load_dotenv()
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
API_KEY = os.getenv('PINECONE_API_KEY')

if not INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME is missing")
if not API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing")

# Type narrowing: assert that INDEX_NAME and API_KEY are non-empty strings
assert isinstance(INDEX_NAME, str) and INDEX_NAME, "INDEX_NAME must be a non-empty string"
assert isinstance(API_KEY, str) and API_KEY, "API_KEY must be a non-empty string"

EMBED_DIMENSION = 1536        
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

class PineconeRetriever:
    def __init__(self):
        self.pc = Pinecone(api_key=API_KEY)

        self._ensure_index_exists()
        self.index = self.pc.Index(INDEX_NAME)  # type: ignore

    def _ensure_index_exists(self):
        existing_indexes = [i["name"] for i in self.pc.list_indexes()]

        if INDEX_NAME not in existing_indexes:
            print(f"[Pinecone] Creating index: {INDEX_NAME}")

            self.pc.create_index(
                name=INDEX_NAME,  # type: ignore
                dimension=EMBED_DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud=CLOUD,
                    region=REGION,
                ),
            )

            # Wait until index is ready
            self._wait_for_index()

        else:
            print(f"[Pinecone] Index '{INDEX_NAME}' found")

    def _wait_for_index(self, timeout: int = 120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.pc.describe_index(INDEX_NAME)  # type: ignore
                print("[Pinecone] Index is ready")
                return
            except Exception:
                time.sleep(3)

        raise TimeoutError("Pinecone index creation timed out")

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> List[Dict[str, Any]]:

        query_vector = embed_query(query)

        if len(query_vector) != EMBED_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"{len(query_vector)} != {EMBED_DIMENSION}"
            )

        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )

        context = []

        for match in result["matches"]: #type: ignore
            metadata = match.get("metadata", {})

            context.append({
                "text": (
                    metadata.get("text")
                    or metadata.get("answer")
                    or metadata.get("content")
                    or ""
                ),
                "question": metadata.get("question", ""),
                "animal": metadata.get("animal", ""),
                "source": metadata.get("source", ""),
                "score": match.get("score", 0.0),
            })

        return context
        

# ------------------------------------------------------------------
# SINGLETON ACCESS
# ------------------------------------------------------------------
_retriever = PineconeRetriever()


def retrieve_context(
    query: str,
    top_k: int = 5,
    namespace = None,
):
    return _retriever.retrieve_context(query, top_k, namespace)

