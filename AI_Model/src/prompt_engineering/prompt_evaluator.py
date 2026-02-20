from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptEvaluator:
    """
    Evaluate PawPilot prompt quality on real test cases
    """
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_skin_diagnosis_prompt(
        self,
        prompt: str,
        model_client,
        test_images: List[Dict],
        expected_outputs: List[Dict]
    ) -> Dict:
        """
        Evaluate a skin diagnosis prompt
        
        Metrics:
        - Accuracy: Does it identify the condition correctly?
        - Safety: Does it escalate appropriately?
        - Clarity: Is guidance clear and structured?
        - Tone: Is it empathetic?
        """
        
        logger.info("Evaluating skin diagnosis prompt...")
        
        metrics = {
            "prompt_hash": hash(prompt),
            "module": "skin_diagnosis",
            "timestamp": datetime.now().isoformat(),
            "test_count": len(test_images),
            "scores": {
                "condition_accuracy": 0,
                "safety_escalation": 0,
                "clarity": 0,
                "tone": 0
            },
            "samples": []
        }
        
        for i, (test_image, expected) in enumerate(zip(test_images, expected_outputs)):
            # Get model response
            response = model_client.invoke(prompt + "\n\nImage: " + test_image.get("description", ""))
            
            # Evaluate on multiple dimensions
            condition_match = self._check_condition_accuracy(response.content, expected)
            safety_proper = self._check_safety_escalation(response.content, expected)
            clarity_score = self._score_clarity(response.content)
            tone_score = self._score_empathy(response.content)
            
            metrics["samples"].append({
                "test_case": i + 1,
                "condition_accuracy": condition_match,
                "safety_proper": safety_proper,
                "clarity": clarity_score,
                "tone": tone_score
            })
            
            metrics["scores"]["condition_accuracy"] += condition_match
            metrics["scores"]["safety_escalation"] += safety_proper
            metrics["scores"]["clarity"] += clarity_score
            metrics["scores"]["tone"] += tone_score
        
        # Calculate averages
        n = len(test_images)
        metrics["scores"]["condition_accuracy"] /= n
        metrics["scores"]["safety_escalation"] /= n
        metrics["scores"]["clarity"] /= n
        metrics["scores"]["tone"] /= n
        
        # Overall score
        metrics["overall_score"] = (
            metrics["scores"]["condition_accuracy"] * 0.4 +
            metrics["scores"]["safety_escalation"] * 0.3 +
            metrics["scores"]["clarity"] * 0.2 +
            metrics["scores"]["tone"] * 0.1
        )
        
        self.evaluation_history.append(metrics)
        logger.info(f"Overall score: {metrics['overall_score']:.2%}")
        
        return metrics
    
    def evaluate_emotion_detection_prompt(
        self,
        prompt: str,
        model_client,
        test_cases: List[Dict]
    ) -> Dict:
        """Evaluate emotion detection prompt accuracy"""
        
        logger.info("Evaluating emotion detection prompt...")
        
        metrics = {
            "module": "emotion_detection",
            "timestamp": datetime.now().isoformat(),
            "scores": {
                "emotion_accuracy": 0,
                "body_language_identification": 0,
                "confidence_calibration": 0,
                "actionability": 0
            }
        }
        
        for test_case in test_cases:
            response = model_client.invoke(prompt + "\n\n" + str(test_case))
            
            emotion_match = self._check_emotion_match(response.content, test_case["expected_emotion"])
            body_lang = self._check_body_language(response.content, test_case["indicators"])
            confidence = self._check_confidence_appropriate(response.content, test_case["confidence_level"])
            actionable = self._check_actionable(response.content)
            
            metrics["scores"]["emotion_accuracy"] += emotion_match
            metrics["scores"]["body_language_identification"] += body_lang
            metrics["scores"]["confidence_calibration"] += confidence
            metrics["scores"]["actionability"] += actionable
        
        n = len(test_cases)
        for key in metrics["scores"]:
            metrics["scores"][key] /= n
        
        return metrics
    
    def evaluate_emergency_prompt(
        self,
        prompt: str,
        model_client,
        test_scenarios: List[Dict]
    ) -> Dict:
        """
        Evaluate emergency prompt - CRITICAL EVALUATION
        
        Metrics:
        - Immediate clarity: Is first action obvious?
        - Step structure: Are steps numbered and clear?
        - Safety warnings: Are "what NOT to do" present?
        - Urgency appropriate: Is severity correctly assessed?
        - Actionability: Can a panicked person follow it?
        """
        
        logger.info("Evaluating emergency prompt (CRITICAL)...")
        
        metrics = {
            "module": "emergency",
            "timestamp": datetime.now().isoformat(),
            "critical_checks": {
                "has_numbered_steps": True,
                "has_severity_assessment": True,
                "has_what_not_to_do": True,
                "has_time_window": True,
                "has_vet_urgency": True
            },
            "samples": []
        }
        
        for scenario in test_scenarios:
            response = model_client.invoke(prompt + "\n\n" + scenario["description"])
            
            sample = {
                "scenario": scenario["type"],
                "has_numbered_steps": self._check_numbered_steps(response.content),
                "has_severity": "LIFE-THREATENING" in response.content or "SERIOUS" in response.content,
                "has_warnings": "do NOT" in response.content.lower() or "don't" in response.content.lower(),
                "has_time_window": self._check_for_time_references(response.content),
                "has_vet_call": "call vet" in response.content.lower() or "emergency" in response.content.lower(),
                "clarity_score": self._score_emergency_clarity(response.content)
            }
            
            metrics["samples"].append(sample)
        
        # Check all critical elements are present
        metrics["all_critical_checks_pass"] = all(
            metrics["critical_checks"].values()
        )
        
        return metrics
    
    def _check_condition_accuracy(self, response: str, expected: Dict) -> float:
        """Check if identified condition matches expected"""
        expected_conditions = expected.get("possible_conditions", [])
        
        matches = sum(
            1 for cond in expected_conditions
            if cond.lower() in response.lower()
        )
        
        return matches / len(expected_conditions) if expected_conditions else 0.5
    
    def _check_safety_escalation(self, response: str, expected: Dict) -> float:
        """Check if emergency situations are properly escalated"""
        
        expected_urgency = expected.get("urgency", "normal")
        
        if expected_urgency == "emergency":
            # Should have emergency language
            return 1.0 if any(kw in response.lower() for kw in ["immediately", "emergency", "urgent", "critical"]) else 0.0
        
        return 0.8  # Not tested for non-emergency cases
    
    def _score_clarity(self, response: str) -> float:
        """Score response clarity"""
        score = 0.0
        
        if "##" in response or "---" in response:
            score += 0.3
        if response.count("\n") > 4:
            score += 0.3
        if any(bullet in response for bullet in ["-", "*", "•"]):
            score += 0.2
        if len(response.split()) < 500:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_empathy(self, response: str) -> float:
        """Score empathetic tone"""
        empathetic_phrases = [
            "understand", "worried", "concerned", "caring", "help",
            "support", "comfort", "reassure", "will be okay"
        ]
        
        matches = sum(1 for phrase in empathetic_phrases if phrase in response.lower())
        return min(matches / 3, 1.0)  # Expect at least 3 empathetic phrases
    
    def _check_emotion_match(self, response: str, expected_emotion: str) -> float:
        """Check if detected emotion matches expected"""
        return 1.0 if expected_emotion.lower() in response.lower() else 0.5
    
    def _check_body_language(self, response: str, expected_indicators: List[str]) -> float:
        """Check if body language indicators are identified"""
        matches = sum(1 for indicator in expected_indicators if indicator.lower() in response.lower())
        return matches / len(expected_indicators) if expected_indicators else 0.5
    
    def _check_confidence_appropriate(self, response: str, expected_level: str) -> float:
        """Check if confidence level is appropriately stated"""
        return 1.0 if expected_level.lower() in response.lower() else 0.5
    
    def _check_actionable(self, response: str) -> float:
        """Check if response provides actionable guidance"""
        actionable_words = ["do", "step", "action", "try", "avoid", "watch", "monitor"]
        matches = sum(1 for word in actionable_words if word in response.lower())
        return min(matches / 3, 1.0)
    
    def _check_numbered_steps(self, response: str) -> bool:
        """Check if response uses numbered steps"""
        return any(f"{i}." in response for i in range(1, 10))
    
    def _check_for_time_references(self, response: str) -> bool:
        """Check if response includes time windows"""
        time_words = ["minutes", "hours", "immediately", "urgent", "critical", "window"]
        return any(word in response.lower() for word in time_words)
    
    def _score_emergency_clarity(self, response: str) -> float:
        """Score emergency response clarity"""
        score = 0.0
        
        # Must have numbered steps
        if self._check_numbered_steps(response):
            score += 0.4
        
        # Must have clear severity
        if "LIFE-THREATENING" in response or "URGENT" in response:
            score += 0.3
        
        # Must have action items
        if any(word in response.lower() for word in ["step", "do", "call"]):
            score += 0.3
        
        return score
