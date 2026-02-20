from gradio_client import Client, handle_file
from PIL import Image
import io
import tempfile
import os
import logging
logger = logging.getLogger(__name__)
client = Client("Aakashdhakse007/Aakash_Model")
def _load_image(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if hasattr(image_input, "read"):
        return Image.open(image_input).convert("RGB")
    return Image.open(image_input).convert("RGB")


def predict(image_path):
    imgs= []
    for img in image_path:
        img = _load_image(img)
        img = img.resize((224, 224))
        imgs.append(img)
  
    try:
        result = []
        for img in imgs:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(tmp.name, "JPEG")
            response = client.predict(handle_file(tmp.name))
            response = result_parser(response)
            result.append(response)
            logger.info(f"Raw Gradio response: {response} (type: {type(response).__name__})")
        
        return result
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def _normalize_images(images):
    if images is None:
        return []
    if not isinstance(images, list):
        return [images]

    normalized = []
    for item in images:
        if isinstance(item, list):
            normalized.extend(item)
        else:
            normalized.append(item)
    return normalized

def result_parser(result):
    result = result.replace("Prediction: ", "")
    result = result.replace("(Confidence: ", "")
    result = result.replace(")", "")
    result = result.replace("%", "")
    result = result.split()
    result[1] = float(result[1]) / 100.0
    return {
        "label": result[0],
        "confidence": result[1]
    } 
def predict_poop_vomit(images):
    images = _normalize_images(images)
    result = predict(images)
    return max(result, key=lambda x: x['confidence']) if result else {"label": "Unknown", "confidence": 0.0}
