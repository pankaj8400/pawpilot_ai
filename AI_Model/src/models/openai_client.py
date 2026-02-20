from openai import OpenAI
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config.constants as config
def _base_model():
    client = OpenAI(
        api_key = config.OPENAIAPI 
    )

    return client