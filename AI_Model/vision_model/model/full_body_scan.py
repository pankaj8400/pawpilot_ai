import google.genai as genai
from google.genai import types
import base64
import re
import os
from dotenv import load_dotenv
from AI_Model.vision_model.utils.load_images import LoadImages
load_dotenv()

def markdown_bold_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
def base64_encode_image(image_path):
    # Accepts either a file-like object, a file path (str), or a PIL Image
    if hasattr(image_path, 'read'):
        return base64.b64encode(image_path.read()).decode('utf-8')
    elif isinstance(image_path, str):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    elif hasattr(image_path, 'save'):  # PIL Image
        from io import BytesIO
        # Convert to RGB if necessary (JPEG doesn't support alpha channels)
        if image_path.mode in ('RGBA', 'LA', 'P'):
            image_path = image_path.convert('RGB')
        buffer = BytesIO()
        image_path.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    else:
        raise TypeError("image_path must be a file-like object, a file path string, or a PIL Image")
def create_content(prompt, images):
    parts = [types.Part(text=prompt)]
    for img in images:
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(base64_encode_image(img))
                )
            )
        )
    return types.Content(parts=parts)
def chatbot_full_body_scan(user_query, images):
    # images is a list of InMemoryUploadedFile objects
    client = genai.Client(http_options={'api_version': 'v1alpha'}, api_key=os.getenv("GEMINI_API"))
    prompt = f"""
           Suppose you are a veterinary expert specializing in dog and cat health assessments, you have been provided with multiple images of a dog from different angles. Your task is to analyze these images to estimate the dog's weight range and overall body condition.
           provide output in more human friendly way, add some emojies according to heading and answer for each point in detail. 
           and if images are not of different animals then create a single report for the dog in the images, not a report for each image.
           Task:
           Estimate the dog's weight range (not exact weight) using all provided images.
           
           Instructions:
           - Use only the provided images
           - Identify the breed or closest visual match (confirm or note deviations)
           - Estimate adult size category (toy / small / medium / large) 
           - Assess body condition (underweight / ideal / overweight)
           - Assign a body condition score (1 to 9)
           - Use visual cues such as body proportions, limb thickness, chest width, waist definition, and fat coverage
           - Do not assume measurements that are not visible
           - Clearly state assumptions and uncertainty

           Output format (strict):
           - Breed (or closest match):
           - Estimated adult size category:
           - Body condition score (1 to 9):
           - Estimated weight range (kg):
           - Confidence level (low / medium / high):
           - Obesity/underweight:
           - Muscle Loss :
           - Limping or joint stifness :
           - Senior posture issue : 
           - Coat quality issues :
           - anxiety/illness posture :
           - Reasoning based on visible features:
           
           """
    loader = LoadImages()
    images = loader.image_loader(strategy="PIL", image_paths=images)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=create_content(prompt, images)
    )
    # Extract the text from the response object
    reply_text = response.text if hasattr(response, 'text') else str(response)
    reply_text = markdown_bold_to_html(reply_text)
    return reply_text

if __name__ == "__main__":
    # Example usage
    images = ["AI_Model/vision_model/data/full body scan/chihuahua/back side chihuahua.JPG",'AI_Model/vision_model/data/full body scan/chihuahua/front side chihuahua.JPG','AI_Model/vision_model/data/full body scan/chihuahua/left side angle chihuahua.JPG','AI_Model/vision_model/data/full body scan/chihuahua/right side angle chihuahua.JPG']
    user_query = "Estimate the dog's weight range and overall body condition using the provided images."
    reply = chatbot_full_body_scan(user_query, images)
    print(reply)
