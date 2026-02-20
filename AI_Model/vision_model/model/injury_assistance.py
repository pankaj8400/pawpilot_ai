from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from dotenv import load_dotenv
from typing import Any

load_dotenv()


def base64_encode_image(image_path):
    image_path.seek(0)
    return base64.b64encode(image_path.read()).decode('utf-8')

def create_message(prompt, image_list):
    content: list[str | dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in image_list:
        img.seek(0)
        base64_image = base64.b64encode(img.read()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    return HumanMessage(content=content)

def chatbot_injury_assistance(user_query, image_path):
    client = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    prompt = f"""You are a medical image analysis assistant...
Analyze the injury image and extract:
1. Injury type
2. Body location
3. severity
4. caused_by
5. species
Return ONLY a JSON object with keys:
- injury_type
- body_location
- severity
- species
- caused_by

User's concern: {user_query}
"""

    message = create_message(prompt, image_path)
    result = client.invoke([message])
    return result