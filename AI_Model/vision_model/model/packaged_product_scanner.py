from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Add project root to sys.path
from AI_Model.vision_model.utils.load_images import MessageLoader, LoadImages
from PIL import Image
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAIAPI")
client = OpenAI(api_key=OPENAI_API_KEY)


def _normalize_image_inputs(image_input):
    if isinstance(image_input, list):
        inputs = image_input
    else:
        inputs = [image_input]

    if not inputs:
        raise ValueError("No image input provided")

    image_loader = LoadImages()
    normalized = []
    for item in inputs:
        if isinstance(item, Image.Image):
            normalized.extend(image_loader.image_to_data_url([item]))
        elif isinstance(item, str):
            if item.startswith("http://") or item.startswith("https://"):
                normalized.append(item)
            else:
                normalized.extend(image_loader.image_to_data_url([item]))
        else:
            raise ValueError(f"Unsupported image input type: {type(item)}")

    return normalized


def extract_text_from_image(image_input):
    """Extract text from an image using OpenAI gpt-4o-mini."""
    image_base64 = _normalize_image_inputs(image_input)
    vision_prompt = """
    You are a vision-based text extraction system for pet products.
    
    Read ALL visible text from the pet product package image.
    
    STEP 1 - Classify the product CAREFULLY:
    
    PRODUCT_TYPE must be one of:
    - "treat": Snacks, treats, chews, training treats, freeze-dried treats, single-ingredient treats (NOT complete meals)
    - "food": Complete meals, kibble, wet food, canned food (provides complete nutrition)
    - "topical": Shampoo, conditioner, spray, paw cream, balm, ear cleaner
    - "supplement": Vitamins, probiotics, joint support pills
    - "litter": Cat litter, bedding material
    - "accessory": Toys, bowls, collars
    
    CRITICAL: If the product says "treat", "snack", or is a single-ingredient item like dried fish/meat, classify as "treat" NOT "food".
    
    PET_TYPE must be one of: dog, cat, both, unknown
    
    STEP 2 - Extract BRAND carefully:
    - Look for company name/logo at top of package
    - Common brands: HUFT, Heads Up For Tails, Pedigree, Whiskas, Royal Canin, etc.
    - If brand is not visible, write "UNREADABLE"
    
    STEP 3 - Extract all other visible text.
    
    STRICT RULES:
    - Do NOT judge health or safety.
    - Do NOT guess unreadable text.
    - Extract nutrition values EXACTLY as shown (e.g., "74.72 g" not "74g")
    
    Output format (plain text):
    PRODUCT_TYPE:
    PET_TYPE:
    PRODUCT_NAME:
    BRAND:
    INGREDIENTS:
    ACTIVE_INGREDIENTS:
    NUTRITION:
    CLAIMS:
    WARNINGS:
    USAGE_INSTRUCTIONS:
    """
    
    from openai.types.chat import ChatCompletionUserMessageParam
    loader = MessageLoader()
    message_dicts = loader.LoadMessages("gpt-4o-mini", vision_prompt, image_base64)
    # Convert dicts to ChatCompletionUserMessageParam objects if needed
    messages = []
    for msg in message_dicts:
        messages.append(ChatCompletionUserMessageParam(**msg))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI Vision API request failed: {e}")


parser_prompt = """
You are a strict data parsing system for pet products.

Convert the input into VALID JSON only.
Do NOT add explanations.
Do NOT infer missing values.

Schema:
{
  "product_info": {
     "name": null,
     "brand": null,
     "type": null,
     "pet_type": null
  },
  "product_type": null,
  "pet_type": null,
  "product_name": null,
  "brand": null,
  "ingredients": [],
  "active_ingredients": [],
  "nutrition": {
    "protein_pct": null,
    "fat_pct": null,
    "fiber_pct": null,
    "moisture_pct": null
  },
  "claims": [],
  "warnings": [],
  "usage_instructions": null
}

NOTES:
- product_type must be one of: "treat", "food", "topical", "supplement", "litter", "accessory"
- "treat" = snacks, chews, training treats, freeze-dried treats (NOT complete meals)
- "food" = complete meals, kibble, wet food
- pet_type must be one of: "dog", "cat", "both", "unknown"
- For treats, nutrition values may be per 100g or per serving

Input:
"""


def parse_to_json(raw_text):
    """Parse raw text into structured JSON using OpenAI gpt-4o-mini."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": parser_prompt + "\n\n" + raw_text
                }
            ],
            response_format={"type": "json_object"} # Force valid JSON
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI Parsing API request failed: {e}")


def process_food_image(image_path):
    """
    Main pipeline function: processes an image and returns structured JSON.
    
    Args:
        image_path: URL or local file path to the food package image
        
    Returns:
        dict: Structured data with product_name, brand, ingredients, nutrition, etc.
              PLUS extraction_confidence and missing_fields
    """
    raw_text = ""
    # Step 1: Extract raw text from image
    images = image_path if isinstance(image_path, list) else [image_path]
    for img in images:
        extracted = extract_text_from_image(img)
        if extracted:
            raw_text += extracted
    
    # Step 2: Parse raw text to JSON string
    json_string = parse_to_json(raw_text)
    
    # Step 3: Convert JSON string to Python dict
    # Clean up potential markdown code blocks from LLM response
    if json_string is None:
        raise ValueError("Parsing returned None instead of a JSON string.")
    json_string = json_string.strip()
    if json_string.startswith("```json"):
        json_string = json_string[7:]
    if json_string.startswith("```"):
        json_string = json_string[3:]
    if json_string.endswith("```"):
        json_string = json_string[:-3]
    
    result = json.loads(json_string.strip())
    
    # Step 4: Calculate extraction confidence and track missing fields
    critical_fields = ['product_type', 'product_name', 'brand', 'ingredients']
    optional_fields = ['nutrition', 'active_ingredients']
    
    missing_fields = []
    field_scores = []
    
    # Check critical fields
    for field in critical_fields:
        value = result.get(field)
        if value is None or value == "" or value == [] or value == "unknown" or value == "UNREADABLE":
            missing_fields.append(field)
            field_scores.append(0.0)
        else:
            field_scores.append(1.0)
    
    # Check optional fields (lower weight)
    for field in optional_fields:
        value = result.get(field)
        if value and value != [] and value != {}:
            field_scores.append(0.5)  # Bonus for having optional data
    
    # Calculate confidence (0-1 scale)
    if len(field_scores) > 0:
        extraction_confidence = sum(field_scores) / (len(critical_fields) + 0.5 * len(optional_fields))
    else:
        extraction_confidence = 0.0
    
    # Clamp to 0-1
    extraction_confidence = max(0.0, min(1.0, extraction_confidence))
    
    # Add metadata to result
    result['extraction_confidence'] = round(extraction_confidence, 2)
    result['missing_fields'] = missing_fields
    
    return result


if __name__ == "__main__":
    image_path = "https://headsupfortails.com/cdn/shop/files/Sara_sWholesome-Lamb_Apple2copy_d823b924-8034-4b32-8dd4-35dff57736f1.jpg?v=1760522131&width=713"

    try:
        result = process_food_image(image_path)
        print("STRUCTURED OUTPUT:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")


