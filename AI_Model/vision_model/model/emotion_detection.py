import google.genai as genai
from google.genai import types
import base64
import re
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()

def markdown_bold_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

def base64_encode_image(image_input):
    """
    Convert image to base64. Handles PIL Images, file-like objects, and file paths.
    """
    # If it's a PIL Image
    if isinstance(image_input, Image.Image):
        buffered = BytesIO()
        image_input.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # If it's a file-like object (FastAPI UploadFile, InMemoryUploadedFile, etc.)
    if hasattr(image_input, 'seek') and hasattr(image_input, 'file'):
        image_input.seek(0)
        return base64.b64encode(image_input.file.read()).decode('utf-8')
    
    # If it's a file-like object with direct read()
    if hasattr(image_input, 'seek') and hasattr(image_input, 'read'):
        image_input.seek(0)
        return base64.b64encode(image_input.read()).decode('utf-8')
    
    # If it's a file path string
    if isinstance(image_input, str):
        with open(image_input, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    raise ValueError(f"Unsupported image type: {type(image_input)}")

def create_content(prompt, images):
    parts = [types.Part(text=prompt)]
    for img in images:
        # Encode the image to base64
        encoded = base64_encode_image(img)
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(encoded)
                )
            )
        )
    return types.Content(parts=parts)

def chatbot_emotion_detection(user_query, images):
    """
    Detect pet emotion from images using Gemini API.
    """
    client = genai.Client(http_options={'api_version': 'v1alpha'}, api_key=os.getenv("GEMINI_API"))
    prompt = f'''You are a Veterinarian AI and you have to detect the emotion of the pet in the image and give suggestions to the owner accordingly. if multiple images are provided and predicted emotion for each image is same then provide a single response otherwise provide response for each image separately.
                by noticing the facial expressions and body language of the dog in the image.
                after that give a brief explanation of 
                Your response should be structured in the following way:
                
                1. Emotional meaning
                2. Training Tips
                3. Comforting actions
                4. Environmental adjustments
                
                User Query: {user_query}'''
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=create_content(prompt, images)
    )
    # Extract the text from the response object
    reply_text = response.text if hasattr(response, 'text') else str(response)
    reply_text = markdown_bold_to_html(reply_text)
    return reply_text

