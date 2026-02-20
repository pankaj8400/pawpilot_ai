import torch
class Aggregation:
    def aggregate_model_predictions(self, all_predictions, all_probs, id2label, device=None): 
        try:
            result1 = self.aggregate_by_voting(all_predictions)
            result2 = self.aggregate_by_confidence(all_predictions)
            result3 = self.aggregate_by_ensemble(all_probs, id2label)
            result4 = self.aggregate_by_weighted(all_probs, id2label, all_predictions, device)
        
            # Collect all predicted labels from the aggregation results
            labels = [
                result1["label"],
                result2["label"],
                result3["label"],
                result4["label"]
            ]
            # Count occurrences of each label
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            # Find the most common label and its count
            most_common_label = max(label_counts, key=lambda k: label_counts[k])
            count = label_counts[most_common_label]
            return {
                "label": most_common_label,
                "confidence": count / 4.0 
            }
        except:
            return {
                "label": None,
                "confidence": 0
            }
        
    def aggregate_by_voting(self,predictions):
        """Majority voting: most common prediction wins"""
        
        try:
            labels = [p["label"] for p in predictions]
            label_counts = {}
            
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        except:
            return {
                "label": None,
                "confidence": 0,
                "votes": {}
            }
        
        winning_label = max(label_counts, key=lambda k: label_counts[k])
        confidence = label_counts[winning_label] / len(predictions)
        
        return {
            "label": winning_label,
            "confidence": confidence,
            "votes": label_counts
        }
    
    
    def aggregate_by_confidence(self,predictions):
        """Pick prediction with highest confidence"""
        try:
            best = max(predictions, key=lambda x: x["confidence"])
        except:
            return {
                "label": None,
                "confidence": 0,
                "method_note": "No valid predictions"
            }
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "method_note": "Highest confidence among all angles"
        }
    
    
    def aggregate_by_ensemble(self, all_probs, id2label):
        """Average probability across all images"""
        try:
            avg_probs = torch.stack(all_probs).mean(dim=0)
            top_prob, top_id = avg_probs.topk(3, dim=1)
        except:
            return {
                "label": None,
                "confidence": 0,
                "top_3": [],
                "method_note": "Failed to aggregate probabilities"
            }
        
        return {
            "label": id2label[top_id[0][0].item()],
            "confidence": top_prob[0][0].item(),
            "top_3": [
                {
                    "label": id2label[top_id[0][i].item()],
                    "confidence": top_prob[0][i].item()
                }
                for i in range(3)
            ],
            "method_note": "Averaged probabilities across all angles"
        }
    
    
    def aggregate_by_weighted(self, all_probs, id2label, predictions, device=None):
        """Weight average by individual prediction confidence"""
        try:
            
            if device is None:
                device = all_probs[0].device
            
            weights = torch.tensor([p["confidence"] for p in predictions], device=device)
            weights = weights / weights.sum()
            
            weighted_probs = torch.stack(all_probs) * weights.unsqueeze(1).unsqueeze(2)
            avg_probs = weighted_probs.sum(dim=0)
            
            top_prob, top_id = avg_probs.topk(1, dim=1)
            
            return {
                "label": id2label[top_id[0].item()],
                "confidence": top_prob[0].item(),
                "method_note": "Weighted average based on individual confidences"
            }
        except:
            return {
                "label": None,
                "confidence": 0,
                "method_note": "Failed to compute weighted average"
            }
#        
#if __name__ == '__main__':
#    # Example usage
#    aggregator = Aggregation()
#    sample_predictions = [
#        {"label": "disease_A", "confidence": 0.8},
#        {"label": "disease_B", "confidence": 0.6},
#        {"label": "disease_A", "confidence": 0.9}
#    ]
#    sample_probs = [
#        torch.tensor([[0.8, 0.2]]),
#        torch.tensor([[0.4, 0.6]]),
#        torch.tensor([[0.9, 0.1]])
#    ]
#    id2label = {0: "disease_A", 1: "disease_B"}
#    
#    result = aggregator.aggregate_model_predictions(sample_predictions, sample_probs, id2label)
#    print("Aggregated Result:", result)