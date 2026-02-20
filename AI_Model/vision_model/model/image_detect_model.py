from AI_Model.vision_model.utils.load_images import LoadImages , MessageLoader
import requests
from dotenv import load_dotenv
import os
load_dotenv()
import requests
def call_nvdia(image_path, prompt="What is in this image?"):
    Loader = LoadImages()
    base64_images = Loader.image_to_data_url(image_path)
    message_loader = MessageLoader()
    message = message_loader.LoadMessages("nvidia/nemotron-nano-12b-v2-vl:free", prompt, base64_images)
    # First API call with reasoning
    response = requests.post(
      url=str(os.getenv('OPENROUTER_API_URL')),
      headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
      },
      json={
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "messages": message,
        "reasoning": {"enabled": True}
      }
    )
    response = response.json()
    response = response
    return response
