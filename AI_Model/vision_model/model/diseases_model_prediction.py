import torch
import torch.nn as nn
import traceback
import clip
from AI_Model.vision_model.utils.method_aggregation import Aggregation
from AI_Model.vision_model.utils.load_images import LoadImages
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_path = "AI_Model/vision_model/model/models/diseases_clip_classifier.pth" 
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval() 
    checkpoint = torch.load(load_path, map_location=device)
    label2id = checkpoint["label2id"]
    id2label = checkpoint["id2label"]
    classifier_weights = checkpoint["classifier_state_dict"]
    
    
    input_dim = model.visual.output_dim
    num_classes = len(label2id)
    classifier = nn.Linear(input_dim, num_classes)
    classifier.load_state_dict(classifier_weights)
    classifier.to(device)
    classifier.eval() 
    
    return model, preprocess, classifier, id2label, device


def predict(image_input, preprocess=None, model=None, classifier=None, id2label=None, device=None):
    if preprocess is None or model is None or classifier is None or id2label is None or device is None:
        model, preprocess, classifier, id2label, device = load_model()
    loader = LoadImages()
    images = loader.image_loader('PIL', image_input)
    image_tensor = []
    for img in images:
        image_tensor.append(preprocess(img).unsqueeze(0).to(device))
    image_tensor = torch.cat(image_tensor, dim=0)
    
    # Get predictions for each image
    all_probs = []
    all_predictions = []
    
    for img in image_tensor:
        with torch.no_grad():
            features = model.encode_image(img.unsqueeze(0))
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.float()   
            
            logits = classifier(features)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs)
            
            top_prob, top_id = probs.topk(1, dim=1)
            all_predictions.append({
                "label": id2label[top_id[0].item()],
                "confidence": top_prob[0].item(),
                "label_id": top_id[0].item()
            })

    aggregator = Aggregation()    
    results = aggregator.aggregate_model_predictions(all_predictions, all_probs, id2label, device=device)
    return results


# --- Usage ---
if __name__ == "__main__":
    try:
        model, preprocess, classifier, id2label, device = load_model()
        image_path = ["AI_Model/vision_model/data/eye corneal unclers.jpg","AI_Model/vision_model/data/eye corneal uclers 2.jpg"]

        result = predict(image_path, preprocess, model, classifier, id2label, device)
        print(f"Prediction: {result}")
    except Exception:
        traceback.print_exc()