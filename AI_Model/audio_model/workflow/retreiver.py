from dotenv import load_dotenv
import os
load_dotenv()

from pinecone import Pinecone
from openai import OpenAI
from typing import Dict, List, Union

def decode_results(results) -> Union[Dict, List[Dict]]:
    """
    Decode Pinecone FetchResponse or QueryResponse into clean dictionaries.
    
    Args:
        results: FetchResponse or QueryResponse from Pinecone
        
    Returns:
        dict or list of dicts containing disease information
    """
    # Handle FetchResponse (from index.fetch())
    if hasattr(results, 'vectors') and isinstance(results.vectors, dict):
        decoded = []
        for vector_id, vector in results.vectors.items():
            if vector.metadata:
                decoded.append({
                    'id': vector_id,
                    'metadata': vector.metadata
                })
        # Return single dict if only one result, else list
        return decoded[0] if len(decoded) == 1 else decoded
    
    # Handle QueryResponse (from index.query())
    elif hasattr(results, 'matches') and isinstance(results.matches, list):
        decoded = []
        for match in results.matches:
            decoded.append({
                'id': match.get('id'),
                'score': match.get('score'),
                'metadata': match.get('metadata', {})
            })
        return decoded
    
    # Fallback
    return results


def retrieve_docs(query: str, index_or_hostname : str = "https://poop-and-vomit-6i6jnuf.svc.aped-4627-b74a.pinecone.io"):
    """
    Retrieve documents from Pinecone based on query.
    
    Args:
        query: Search query string
        index_or_hostname: Either a Pinecone index name or hostname URL
        
    Returns:
        Decoded results from Pinecone or empty list if retrieval fails
    """
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API"))
        
        # Validate input
        if not index_or_hostname:
            print(f"Warning: No index or hostname provided, returning empty results")
            return []
        
        # Check if it's a hostname (URL) or index name
        try:
            if index_or_hostname.startswith("http"):
                index = pc.Index(host=index_or_hostname)
            else:
                index = pc.Index(name=index_or_hostname)
        except Exception as e:
            print(f"Error connecting to Pinecone index: {e}")
            return []
        
        # Try exact ID match with various formats
        possible_ids = [
            query.lower(),
            query.replace(" ", "_").lower(), 
            query.replace(" ", "-").lower(),
            query.title().lower(),
            query
        ]
        
        try:
            results = index.fetch(
                ids=possible_ids,
                namespace="__default__"
            )
            
            # If exact ID match found, return it
            if results and hasattr(results, 'vectors') and results.vectors:
                return decode_results(results)
        except Exception as e:
            print(f"ID fetch from Pinecone failed: {e}")
        
        # Fall back to vector similarity search
        try:
            if index_or_hostname != "https://poop-and-vomit-6i6jnuf.svc.aped-4627-b74a.pinecone.io" and index_or_hostname != "https://pet-food-image-analysis-6i6jnuf.svc.aped-4627-b74a.pinecone.io":
                client = OpenAI(api_key=os.getenv("OPENAIAPI"))
                embedding = client.embeddings.create(
                    input=query,
                    model="text-embedding-3-large",
                    dimensions=512
                )
                embedding = embedding.data[0].embedding
            if index_or_hostname == "https://pet-food-image-analysis-6i6jnuf.svc.aped-4627-b74a.pinecone.io":
                client = OpenAI(api_key=os.getenv("OPENAIAPI"))
                embedding_response = client.embeddings.create(
                    input=query,
                    model="text-embedding-3-large",
                    dimensions=1024
                )
                embedding = embedding_response.data[0].embedding
            else:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-mpnet-base-v2')
                embedding = model.encode([query])[0].tolist()
            
            results = index.query(
                namespace="__default__",
                vector=embedding,
                top_k=1,
                include_metadata=True
            )
            
            return decode_results(results)
        except Exception as e:
            print(f"Vector similarity search failed: {e}")
            return []
    
    except Exception as e:
        print(f"Error in retrieve_docs: {e}")
        return []

if __name__ == "__main__":
    query = "alert"
    results = retrieve_docs(query)
    print(results)