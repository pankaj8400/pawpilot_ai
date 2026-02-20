from PIL import Image
import base64
class LoadImages:
    def image_loader(self, strategy,image_paths):
        images = []
        if strategy == "PIL":
            for img in image_paths:
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                else:
                    images.append(Image.open(img).convert("RGB"))
            return images

        elif strategy == 'Base64':
            for img_path in image_paths:
                with open(img_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                    images.append(b64_string)
        elif strategy == 'BytesIO':
            import io
            for img_bytes in image_paths:
                if isinstance(img_bytes, bytes):
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                else:
                    raise ValueError("Expected bytes input for BytesIO strategy.")
                
        return images
        
    
    def image_to_data_url(self, image_path):
        result = []
        for img in image_path:
            # Handle PIL Image objects
            if isinstance(img, Image.Image):
                from io import BytesIO
                # Convert to RGB if necessary (JPEG doesn't support alpha channels)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                result.append(f"data:image/jpeg;base64,{b64_string}")
            # Handle file paths
            elif isinstance(img, str):
                with open(img, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                    result.append(f"data:image/jpeg;base64,{b64_string}")
            else:
                raise ValueError(f"Expected PIL Image or file path string, got {type(img)}")
        return result

class MessageLoader:
    def LoadMessages(self,model, query, images):
        base_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe this image in detail and also detect the {query}"},
                ],
            }
        ]
        for img in images:
            if isinstance(img, dict):
                image_url = img
            else:
                image_url = {"url": img}
            base_message[0]["content"].append({"type": "image_url", "image_url": image_url})
        return base_message
    
