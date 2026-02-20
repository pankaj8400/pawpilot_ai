import json
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FewShotBuilder:
    """
    Manage and optimize few-shot examples for PawPilot
    """
    
    def __init__(self, examples_file: str = "src/prompt_engineering/templates/few_shot_examples.json"):
        self.examples_file = examples_file
        self.examples = self._load_examples()
    
    def _load_examples(self) -> Dict:
        """Load few-shot examples from file"""
        try:
            with open(self.examples_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Examples file not found: {self.examples_file}")
            return {}
    
    def get_examples_for_module(
        self,
        module_type: str,
        num_examples: int = 2
    ) -> List[Dict]:
        """
        Get few-shot examples for a specific PawPilot module
        
        Args:
            module_type: "skin_diagnosis", "emotion_detection", "emergency", etc
            num_examples: How many examples to return
        
        Returns:
            List of example dicts with input/output pairs
        """
        
        module_examples = self.examples.get(module_type, {}).get("examples", [])
        
        if not module_examples:
            logger.warning(f"No examples found for {module_type}")
            return []
        
        return module_examples[:num_examples]
    
    def select_relevant_examples(
        self,
        query: str,
        module_type: str,
        num_examples: int = 2
    ) -> List[Dict]:
        """
        Select most relevant few-shot examples based on query similarity
        
        Instead of always using the same examples, pick ones similar to the user's query
        """
        
        module_examples = self.examples.get(module_type, {}).get("examples", [])
        
        if not module_examples:
            return []
        
        # Score examples by similarity to query
        scored = []
        for example in module_examples:
            similarity = self._calculate_similarity(
                query,
                example.get("input", "")
            )
            scored.append((example, similarity))
        
        # Sort by similarity and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex[0] for ex in scored[:num_examples]]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union
    
    def format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """
        Format few-shot examples into prompt-ready string
        
        Converts list of examples into readable prompt format
        """
        
        formatted = ""
        
        for i, example in enumerate(examples, 1):
            formatted += f"\n## Example {i}:\n"
            
            if "input" in example:
                formatted += f"**Input/Scenario:** {example['input']}\n\n"
            
            if "output" in example:
                formatted += f"**Expected Output:**\n{example['output']}\n"
            
            if "reasoning" in example:
                formatted += f"\n**Reasoning:** {example['reasoning']}\n"
        
        return formatted
    
    def add_example_from_feedback(
        self,
        module_type: str,
        user_input: str,
        ai_output: str,
        user_rating: int,
        reasoning: str = ""
    ):
        """
        Add new example from successful user interaction
        
        Continuously improve few-shot examples based on real usage
        """
        
        if module_type not in self.examples:
            self.examples[module_type] = {"examples": []}
        
        new_example = {
            "input": user_input,
            "output": ai_output,
            "rating": user_rating,
            "confidence": user_rating / 5.0,  # Convert 1-5 rating to 0-1 confidence
            "reasoning": reasoning,
            "added_date": datetime.now().isoformat()
        }
        
        self.examples[module_type]["examples"].append(new_example)
        self._save_examples()
        
        logger.info(f"Added new example to {module_type} (rating: {user_rating}/5)")
    
    def _save_examples(self):
        """Save examples back to file"""
        with open(self.examples_file, 'w') as f:
            json.dump(self.examples, f, indent=2)
    
    def get_high_confidence_examples(
        self,
        module_type: str,
        min_confidence: float = 0.8
    ) -> List[Dict]:
        """Get only high-quality examples (high user ratings)"""
        
        module_examples = self.examples.get(module_type, {}).get("examples", [])
        
        return [
            ex for ex in module_examples
            if ex.get("confidence", 0) >= min_confidence
        ]
