from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import re

#configure the tool
load_dotenv()
search =  TavilySearchResults(max_results=5)

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
    


@tool
def web_search(query:str) -> str:
    """Search the web for the pet health information, product, recalls, and vet advice"""
    
    results = search.invoke(query)

    if not results:
        return "No relevant web results found."

    results = sorted(results, key=lambda x:x.get("score", 0), reverse=True)

    results = results[:3]

    context_blocks = []

    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = clean_text(r.get("content", ))[:1200]

        blocks = f"""
        Source : {title} 
        
        Url : {url}

        content : {content}"""

        context_blocks.append(blocks)

    final_content = "\n\n".join(context_blocks)

    return final_content