import base64
from AI_Model.src.prompt_engineering.food_model_prompts import get_food_vision_prompt, route_food_query
import google.genai as genai
from google.genai import types
import os 
import io
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

def _image_to_base64(image):
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        rgb_image = image.convert("RGB")
        rgb_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(image).decode("utf-8")
    if hasattr(image, "read"):
        if hasattr(image, "seek"):
            image.seek(0)
        return base64.b64encode(image.read()).decode("utf-8")
    raise ValueError(f"Unsupported image type: {type(image)}")

def create_content(prompt, image):
    parts = [types.Part(text=str(prompt))]
    images = image if isinstance(image, list) else [image]
    for img in images:
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(_image_to_base64(img))
                )
            )
        )
    return types.Content(parts=parts)


def chatbot_food_analyzer(user_query, image):
    client = genai.Client(http_options={'api_version': 'v1alpha'}, api_key=os.getenv("GEMINI_API"))
    route_info = route_food_query(user_query)
    prompt = get_food_vision_prompt(route_info.get('vision_context', 'standard'), route_info.get('species', 'unknown'))
    content = create_content(prompt, image)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=content
    )
    reply_text = response.text if hasattr(response, 'text') else str(response)
    return reply_text