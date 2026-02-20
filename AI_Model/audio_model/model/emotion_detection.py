from gradio_client import Client, handle_file
import tempfile

client = Client("pankaj8400/Pankaj")

def predict_emotion(audio_path):
    results = []
    for file in audio_path:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(file, "rb") as f:
            tmp.write(f.read())
        result = client.predict(handle_file(tmp.name))
        results.append(result)
    return results

