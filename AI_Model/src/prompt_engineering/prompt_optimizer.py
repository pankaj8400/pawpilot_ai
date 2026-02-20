import json
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Test and optimize PawPilot prompts using A/B testing and metrics
    """
    
    def __init__(self):
        self.test_results = []
        self.best_prompts = {}
    
    def a_b_test_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        model_client,
        test_cases: List[Dict],
        module_type: str = "skin_diagnosis"
    ) -> Dict:
        """
        A/B test two prompts on the same test cases
        
        Args:
            prompt_a: First prompt variant
            prompt_b: Second prompt variant
            model_client: OpenAI or other LLM client
            test_cases: List of test inputs
            module_type: Which PawPilot module (skin_diagnosis, emotion, etc)
        
        Returns:
            Winner and detailed comparison
        
        Example:
            result = optimizer.a_b_test_prompts(
                prompt_a=prompt_with_context,
                prompt_b=prompt_without_context,
                model_client=openai_client,
                test_cases=test_cases,
                module_type="skin_diagnosis"
            )
            print(f"Winner: {result['winner']} (Score: {result['winner_score']})")
        """
        
        logger.info(f"Starting A/B test for {module_type}...")
        
        # Test prompt A
        results_a = self._test_prompt(prompt_a, model_client, test_cases, "Prompt A")
        
        # Test prompt B
        results_b = self._test_prompt(prompt_b, model_client, test_cases, "Prompt B")
        
        # Compare results
        comparison = {
            "module": module_type,
            "timestamp": datetime.now().isoformat(),
            "prompt_a": {
                "accuracy": results_a["accuracy"],
                "avg_length": results_a["avg_length"],
                "clarity_score": results_a["clarity_score"],
                "cost": results_a["cost"],
                "samples": results_a["samples"]
            },
            "prompt_b": {
                "accuracy": results_b["accuracy"],
                "avg_length": results_b["avg_length"],
                "clarity_score": results_b["clarity_score"],
                "cost": results_b["cost"],
                "samples": results_b["samples"]
            }
        }
        
        # Determine winner
        winner, score = self._determine_winner(results_a, results_b)
        
        comparison["winner"] = winner
        comparison["winner_score"] = score
        comparison["recommendation"] = f"Use {winner} (Score: {score:.2%})"
        
        self.test_results.append(comparison)
        self.best_prompts[module_type] = winner
        
        logger.info(f"A/B test complete. Winner: {winner}")
        
        return comparison
    
    def _test_prompt(
        self,
        prompt: str,
        model_client,
        test_cases: List[Dict],
        prompt_name: str
    ) -> Dict:
        """Test a single prompt on multiple cases"""
        
        results = {
            "name": prompt_name,
            "samples": [],
            "accuracy": 0,
            "avg_length": 0,
            "clarity_score": 0,
            "cost": 0
        }
        
        total_accuracy = 0
        total_length = 0
        total_clarity = 0
        total_tokens = 0
        
        for i, test_case in enumerate(test_cases):
            # Get model response
            full_prompt = prompt + "\n\nINPUT:\n" + test_case["input"]
            
            response = model_client.invoke(full_prompt)
            
            # Score response
            accuracy = self._score_accuracy(response.content, test_case)
            clarity = self._score_clarity(response.content)
            
            sample = {
                "test_case": i + 1,
                "input": test_case["input"],
                "expected": test_case.get("expected", ""),
                "output": response.content,
                "accuracy": accuracy,
                "clarity": clarity,
                "length": len(response.content.split())
            }
            
            results["samples"].append(sample)
            
            total_accuracy += accuracy
            total_length += len(response.content.split())
            total_clarity += clarity
            total_tokens += getattr(response, "tokens", 0)
        
        # Calculate averages
        n = len(test_cases)
        results["accuracy"] = total_accuracy / n if n > 0 else 0
        results["avg_length"] = total_length / n if n > 0 else 0
        results["clarity_score"] = total_clarity / n if n > 0 else 0
        results["cost"] = (total_tokens / 1000) * 0.002  # Rough estimate
        
        return results
    
    def _score_accuracy(self, response: str, test_case: Dict) -> float:
        """Score response accuracy (0-1)"""
        expected = test_case.get("expected", "")
        if not expected:
            return 0.5
        
        # Simple keyword matching
        expected_keywords = set(expected.lower().split())
        response_keywords = set(response.lower().split())
        
        if not expected_keywords:
            return 1.0
        
        overlap = len(expected_keywords & response_keywords)
        return overlap / len(expected_keywords)
    
    def _score_clarity(self, response: str) -> float:
        """Score response clarity (0-1)"""
        score = 0.0
        
        # Check for structure
        has_headers = bool("##" in response or "---" in response)
        has_bullets = bool("-" in response or "*" in response)
        has_newlines = response.count("\n") > 2
        
        if has_headers:
            score += 0.4
        if has_bullets:
            score += 0.3
        if has_newlines:
            score += 0.3
        
        # Penalize if too long
        words = len(response.split())
        if words > 500:
            score *= 0.8
        
        return min(score, 1.0)
    
    def _determine_winner(self, results_a: Dict, results_b: Dict) -> Tuple[str, float]:
        """Determine which prompt performed better"""
        
        # Weighted scoring: accuracy (50%) + clarity (30%) + cost (20%)
        score_a = (
            results_a["accuracy"] * 0.5 +
            results_a["clarity_score"] * 0.3 +
            (1 - (results_a["cost"] / max(results_a["cost"], results_b["cost"]))) * 0.2
        )
        
        score_b = (
            results_b["accuracy"] * 0.5 +
            results_b["clarity_score"] * 0.3 +
            (1 - (results_b["cost"] / max(results_a["cost"], results_b["cost"]))) * 0.2
        )
        
        if score_a > score_b:
            return "Prompt A", score_a
        elif score_b > score_a:
            return "Prompt B", score_b
        else:
            return "Tie", score_a
    
    def optimize_for_module(
        self,
        module_type: str,
        prompt_variants: List[str],
        model_client,
        test_cases: List[Dict]
    ) -> str:
        """
        Find the best prompt from multiple variants
        
        Returns the best performing prompt
        """
        
        logger.info(f"Optimizing {len(prompt_variants)} prompt variants for {module_type}...")
        
        best_prompt = None
        best_score = -1
        results_summary = []
        
        for i, prompt in enumerate(prompt_variants):
            results = self._test_prompt(prompt, model_client, test_cases, f"Variant {i+1}")
            
            score = (
                results["accuracy"] * 0.5 +
                results["clarity_score"] * 0.3
            )
            
            results_summary.append({
                "variant": i + 1,
                "score": score,
                "accuracy": results["accuracy"],
                "clarity": results["clarity_score"]
            })
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        logger.info(f"Best variant: {[r for r in results_summary if r['score'] == best_score][0]['variant']}")
        
        return str(best_prompt)
    
    def save_optimization_results(self, filename: str = "prompt_optimization_results.json"):
        """Save A/B test results for analysis"""
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Optimization results saved to {filename}")

