from typing import List
from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv
class EmbeddingServices:
    """
    Handles embedding genration for query
    """
    load_dotenv()
    def __init__(self):
        api_key = os.getenv('OPENAIAPI')
        self.client=OpenAI(api_key=api_key)
        self.model = os.getenv('EMBEDDING_MODEL')

    def embed_query(self,query:str)->List[float]:

        if not query or not query.strip(): 
            raise ValueError("Question text is empty")

        response=self.client.embeddings.create(
            model=self.model,
            input=query,
            dimensions = 1536
        )
        return response.data[0].embedding

_embeding_service=EmbeddingServices()
def embed_query(query:str)->List[float]:
    return _embeding_service.embed_query(query)
    

