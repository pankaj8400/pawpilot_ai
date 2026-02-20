from gradio_client import Client, handle_file
from PIL import Image
import io
import tempfile

client = Client("Codesutra/parasite-detection")

def _load_image(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if hasattr(image_input, "read"):
        return Image.open(image_input).convert("RGB")
    return Image.open(image_input).convert("RGB")


def predict_parasite(image_path):
    img = _load_image(image_path)
    img = img.resize((224,224))

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG")

    result = client.predict(handle_file(tmp.name))

    return result


def predict_parasites(images):
    result_classes = []
    result_confidence = []
    if len(images) > 1:
        for img in images:
            result = predict_parasite(img)
            result = result.split()
            result[1] = result[1].replace("(","")
            result[1] = result[1].replace(")","")
            result_classes.append(result[0])
            result_confidence.append(float(result[1]))
    else:
        result = predict_parasite(images[0])
        result = result.split()
        result[1] = result[1].replace("(","")
        result[1] = result[1].replace(")","")
        result_classes = result[0]
        result_confidence = float(result[1])
    return result_classes, result_confidence
