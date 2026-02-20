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


def retrieve_docs(query: str, index_or_hostname):
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
                top_k=5,
                include_metadata=True
            )
            
            return decode_results(results)
        except Exception as e:
            print(f"Vector similarity search failed: {e}")
            return []
    
    except Exception as e:
        print(f"Error in retrieve_docs: {e}")
        return []

def create_embeddings(model, text:str) -> List[float]:
    """
    Create embeddings for given text using OpenAI API.
    
    Args:
        text: Input text to embed"""
    try:
        query_vector = model.encode([text])[0].tolist()
        return query_vector
    
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []
def upsert_data(host_name : str, data : List[Dict]):
    """
    Upsert data into Pinecone index.
    
    Args:
        host_name: Pinecone index hostname
        data: List of dictionaries containing 'id', 'metadata', and 'vector' keys
        
    Returns:
        None
    """
    try:
        from sentence_transformers import SentenceTransformer
        pc = Pinecone(api_key=os.getenv("PINECONE_API"))
        index = pc.Index(host=host_name)
        model = SentenceTransformer('all-mpnet-base-v2')
        for item in data:
            vectors = []
            vector_id = create_embeddings(model, str(item.get('class_id')))
            meta_data = {}
            meta_data['primary_system'] = item.get('primary_system', '')
            meta_data['sample_type'] = item.get('sample_type', '')
            meta_data['arousal_level'] = item.get('arousal_level', 0)
            meta_data['valence'] = item.get('valence', '')
            meta_data['urgency_score'] = item.get('urgency_score', 0)
            meta_data['is_pathological'] = item.get('is_pathological', '')
            meta_data['description'] = item.get('description', '')

            if vector_id and meta_data:
                vectors.append({
                    'id': str(item.get('class_id',"")),
                    'values': vector_id,
                    'metadata': meta_data
                })
            index.upsert(vectors=vectors, namespace="__default__")
            
    except Exception as e:
        print(f"Error in upsert_data: {e}")

if __name__ == "__main__":
    import json 
    data = json.load(open("AI_Model/vision_model/data/emotion-detection-audio.json"))
    upsert_data(host_name="https://poop-and-vomit-6i6jnuf.svc.aped-4627-b74a.pinecone.io", data=data)
    print("done")