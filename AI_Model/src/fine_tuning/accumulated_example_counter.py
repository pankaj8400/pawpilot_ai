import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class AccumulatedExamplesCounter:
    """
    Track and count accumulated training examples from user feedback
    
    Purpose:
    - Collect high-quality Q&A pairs from user interactions
    - Count examples to determine when fine-tuning should trigger
    - Manage training data quality
    - Provide analytics on accumulated data
    
    Key Metrics:
    - Total examples accumulated
    - High-quality examples (4-5 stars)
    - Examples by module
    - Examples by quality level
    - Days since last fine-tuning
    """
    
    def __init__(
        self,
        accumulated_file: str = "data/training/accumulated_examples.jsonl",
        threshold: int = 100
    ):
        """
        Initialize the counter
        
        Args:
            accumulated_file: File storing accumulated examples (JSONL)
            threshold: Minimum examples needed for fine-tuning (default: 100)
        """
        
        self.accumulated_file = Path(accumulated_file)
        self.accumulated_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.threshold = threshold
        self.examples = []
        
        # Load existing examples from file
        self.load_examples()
        
        logger.info(f"AccumulatedExamplesCounter initialized")
        logger.info(f"  File: {self.accumulated_file}")
        logger.info(f"  Threshold: {threshold} examples")
        logger.info(f"  Current count: {len(self.examples)} examples")
    
    
    # ====================================================================
    # ADD NEW EXAMPLES (from user feedback)
    # ====================================================================
    
    def add_example(
        self,
        user_query: str,
        ai_response: str,
        user_rating: int,
        module: str,
        pet_id: Optional[str] = None,
        user_id: Optional[str] = None,
        additional_feedback: Optional[str] = None
    ) -> None:
        """
        Add a single training example from user interaction
        
        CALLED AFTER USER PROVIDES FEEDBACK (Node 7)
        
        Args:
            user_query: What the user asked
            ai_response: What PawPilot AI responded
            user_rating: User satisfaction (1-5 stars)
            module: Which PawPilot module (skin_diagnosis, emotion, emergency, etc)
            pet_id: Optional pet identifier
            user_id: Optional user identifier
            additional_feedback: Optional user comments
        
        Example:
            counter.add_example(
                user_query="My dog has a red rash on his paw",
                ai_response="Based on the image, this looks like dermatitis...",
                user_rating=5,
                module="skin_diagnosis",
                pet_id="pet_max_123",
                user_id="user_john_456",
                additional_feedback="Very helpful and accurate!"
            )
        """
        
        logger.info(f"Adding example: rating={user_rating}, module={module}")
        
        try:
            # Create example record with all metadata
            example = {
                "id": f"{user_id}_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "ai_response": ai_response,
                "user_rating": user_rating,  # 1-5 stars
                "confidence": user_rating / 5.0,  # Convert to 0-1 confidence
                "module": module,
                "pet_id": pet_id,
                "user_id": user_id,
                "feedback": additional_feedback,
                "quality": self._determine_quality(user_rating),
                "is_high_quality": user_rating >= 4
            }
            
            # Add to in-memory list
            self.examples.append(example)
            
            # Save to file
            self._save_example_to_file(example)
            
            # Log current count
            current_count = len(self.examples)
            high_quality_count = self.count_high_quality_examples()
            
            logger.info(f"✓ Example added")
            logger.info(f"  Total: {current_count}/{self.threshold}")
            logger.info(f"  High-quality: {high_quality_count}")
            
            # Check if threshold reached
            if current_count >= self.threshold:
                logger.warning(f"""
╔════════════════════════════════════════════════════════════╗
║ 🎯 FINE-TUNING THRESHOLD REACHED!                         ║
║ {current_count} examples accumulated                        ║
║ Ready for fine-tuning job submission                       ║
╚════════════════════════════════════════════════════════════╝
""")
        
        except Exception as e:
            logger.error(f"Failed to add example: {str(e)}", exc_info=True)
    
    
    def _determine_quality(self, rating: int) -> str:
        """
        Determine quality level based on user rating
        
        Returns: "high" | "medium" | "low"
        """
        if rating >= 4:
            return "high"
        elif rating >= 3:
            return "medium"
        else:
            return "low"
    
    
    def _save_example_to_file(self, example: Dict) -> None:
        """Save example to JSONL file for persistence"""
        
        try:
            with open(self.accumulated_file, 'a') as f:
                f.write(json.dumps(example) + '\n')
            logger.debug(f"Example saved to {self.accumulated_file}")
        except Exception as e:
            logger.error(f"Failed to save example to file: {str(e)}")
    
    
    # ====================================================================
    # COUNT EXAMPLES (Main threshold checks)
    # ====================================================================
    
    def count_accumulated_examples(self) -> int:
        """
        Count total accumulated examples
        
        CALLED BY NODE 8 FOR THRESHOLD CHECK
        
        Returns:
            Total number of accumulated examples
        
        Example usage:
            accumulated = counter.count_accumulated_examples()
            if accumulated >= 100:
                # Trigger fine-tuning!
        """
        
        count = len(self.examples)
        logger.info(f"Total accumulated examples: {count}/{self.threshold}")
        return count
    
    
    def count_high_quality_examples(self, min_rating: int = 4) -> int:
        """
        Count only high-quality examples (user rating >= min_rating)
        
        Only examples with 4+ stars should be used for fine-tuning
        
        Args:
            min_rating: Minimum rating to count (default: 4 = 4-5 stars)
        
        Returns:
            Count of high-quality examples
        
        Example:
            high_quality = counter.count_high_quality_examples(min_rating=4)
            if high_quality >= 80:  # 80% of 100 examples
                ready_for_training = True
        """
        
        count = sum(1 for ex in self.examples if ex["user_rating"] >= min_rating)
        percentage = (count / len(self.examples) * 100) if self.examples else 0
        
        logger.info(f"High-quality examples (rating >= {min_rating}): {count} ({percentage:.1f}%)")
        return count
    
    
    def count_by_quality(self) -> Dict[str, int]:
        """
        Count examples grouped by quality level
        
        Returns:
            {
                "high": 85,      # 4-5 stars
                "medium": 10,    # 3 stars
                "low": 5,        # 1-2 stars
                "total": 100
            }
        
        Example:
            quality_breakdown = counter.count_by_quality()
            print(f"High: {quality_breakdown['high']}")  # Output: 85
        """
        
        quality_counts = defaultdict(int)
        
        for example in self.examples:
            quality = example.get("quality", "low")
            quality_counts[quality] += 1
        
        result = {
            "high": quality_counts["high"],
            "medium": quality_counts["medium"],
            "low": quality_counts["low"],
            "total": len(self.examples)
        }
        
        logger.info(f"Quality breakdown: {result}")
        return result
    
    
    def count_by_module(self) -> Dict[str, int]:
        """
        Count examples grouped by PawPilot module
        
        Returns:
            {
                "skin_diagnosis": 35,
                "emotion_detection": 25,
                "emergency": 20,
                "product_safety": 15,
                "behavior": 5,
                "total": 100
            }
        
        Example:
            module_breakdown = counter.count_by_module()
            for module, count in module_breakdown.items():
                print(f"{module}: {count}")
        """
        
        module_counts = defaultdict(int)
        
        for example in self.examples:
            module = example.get("module", "unknown")
            module_counts[module] += 1
        
        result = dict(module_counts)
        result["total"] = len(self.examples)
        
        logger.info(f"Module breakdown: {result}")
        return result
    
    
    def count_by_user(self) -> Dict[str, int]:
        """
        Count examples grouped by user
        
        Useful for understanding which users provide good feedback
        
        Returns:
            {
                "user_123": 15,
                "user_456": 8,
                "anonymous": 77,
                "total": 100
            }
        """
        
        user_counts = defaultdict(int)
        
        for example in self.examples:
            user = example.get("user_id", "anonymous")
            user_counts[user] += 1
        
        result = dict(user_counts)
        result["total"] = len(self.examples)
        
        logger.info(f"User breakdown: {result}")
        return result
    
    
    def count_by_pet(self) -> Dict[str, int]:
        """
        Count examples grouped by pet
        
        Returns:
            {
                "pet_max_123": 25,
                "pet_luna_456": 18,
                "unknown": 57,
                "total": 100
            }
        """
        
        pet_counts = defaultdict(int)
        
        for example in self.examples:
            pet = example.get("pet_id", "unknown")
            pet_counts[pet] += 1
        
        result = dict(pet_counts)
        result["total"] = len(self.examples)
        
        return result
    
    
    def count_since_last_training(self, last_training_date: str) -> int:
        """
        Count examples accumulated since last fine-tuning
        
        Args:
            last_training_date: ISO format datetime (e.g., "2024-01-08T14:30:00Z")
        
        Returns:
            Count of examples since last training
        
        Example:
            new_examples = counter.count_since_last_training("2024-01-08T14:30:00Z")
            if new_examples >= 50:  # Enough new data since last training
                consider_retraining = True
        """
        
        last_training = datetime.fromisoformat(last_training_date.replace('Z', '+00:00'))
        
        new_count = sum(
            1 for ex in self.examples
            if datetime.fromisoformat(ex["timestamp"]) > last_training
        )
        
        logger.info(f"Examples since last training ({last_training_date}): {new_count}")
        return new_count
    
    
    # ====================================================================
    # READY FOR FINE-TUNING? (Threshold checks)
    # ====================================================================
    
    def is_ready_for_fine_tuning(self) -> Dict:
        """
        Check if system is ready for fine-tuning
        
        CALLED BY NODE 8 TO MAKE FINAL DECISION
        
        Checks:
        - Have we accumulated enough examples? (>= 100)
        - Are most examples high quality? (>= 70%)
        - Do we have diverse module coverage?
        
        Returns:
            {
                "ready": True/False,
                "current_count": 100,
                "threshold": 100,
                "high_quality_percentage": 85.0,
                "examples_ok": True,
                "quality_ok": True,
                "diversity_ok": True,
                "reasons": ["All checks passed!"],
                "module_breakdown": {...},
                "quality_breakdown": {...}
            }
        
        Example usage:
            readiness = counter.is_ready_for_fine_tuning()
            if readiness["ready"]:
                # Trigger fine-tuning!
        """
        
        total = len(self.examples)
        high_quality = self.count_high_quality_examples(min_rating=4)
        quality_ratio = (high_quality / total * 100) if total > 0 else 0
        
        reasons = []
        
        # CHECK 1: Enough examples?
        examples_ok = total >= self.threshold
        if not examples_ok:
            reasons.append(f"Need {self.threshold - total} more examples ({total}/{self.threshold})")
        else:
            reasons.append(f"✓ {total} examples (threshold: {self.threshold})")
        
        # CHECK 2: High quality?
        quality_ok = quality_ratio >= 70.0
        if not quality_ok:
            reasons.append(f"Only {quality_ratio:.1f}% high quality (need 70%)")
        else:
            reasons.append(f"✓ {quality_ratio:.1f}% high quality")
        
        # CHECK 3: Module diversity?
        module_counts = self.count_by_module()
        module_counts.pop("total", None)
        
        min_per_module = total // 5 // 2 if total > 0 else 10  # At least 10% per major module
        
        diversity_ok = all(
            count >= min_per_module 
            for module, count in module_counts.items()
        ) if module_counts else False
        
        if not diversity_ok:
            reasons.append(f"Uneven distribution across modules")
        else:
            reasons.append(f"✓ Good module diversity")
        
        # FINAL DECISION
        ready = examples_ok and quality_ok and diversity_ok
        
        logger.info(f"Fine-tuning readiness: {ready}")
        if ready:
            logger.warning(f"✅ SYSTEM READY FOR FINE-TUNING!")
        
        return {
            "ready": ready,
            "current_count": total,
            "threshold": self.threshold,
            "high_quality_count": high_quality,
            "high_quality_percentage": quality_ratio,
            "examples_ok": examples_ok,
            "quality_ok": quality_ok,
            "diversity_ok": diversity_ok,
            "reasons": reasons,
            "module_breakdown": self.count_by_module(),
            "quality_breakdown": self.count_by_quality()
        }
    
    
    # ====================================================================
    # LOAD & MANAGE EXAMPLES
    # ====================================================================
    
    def load_examples(self) -> None:
        """Load previously saved examples from JSONL file"""
        
        if not self.accumulated_file.exists():
            logger.info(f"No accumulated examples file found: {self.accumulated_file}")
            return
        
        try:
            with open(self.accumulated_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        example = json.loads(line.strip())
                        self.examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse line {line_num}: {str(e)}")
            
            logger.info(f"Loaded {len(self.examples)} accumulated examples from file")
        
        except Exception as e:
            logger.error(f"Failed to load examples: {str(e)}")
    
    
    def get_all_examples(self) -> List[Dict]:
        """
        Get all accumulated examples
        
        Returns:
            List of all example dictionaries
        """
        return self.examples.copy()
    
    
    def get_examples_for_training(self) -> List[Dict]:
        """
        Get examples suitable for fine-tuning
        
        Filters: Only high-quality examples (rating >= 4)
        
        Returns:
            List of training-ready examples
        
        Example:
            training_data = counter.get_examples_for_training()
            # Will return only 4-5 star examples
        """
        
        training_examples = [
            ex for ex in self.examples 
            if ex["user_rating"] >= 4
        ]
        
        logger.info(f"Training examples (high quality only): {len(training_examples)}")
        return training_examples
    
    
    def get_examples_by_module(self, module: str) -> List[Dict]:
        """
        Get all examples for a specific module
        
        Args:
            module: Module name (skin_diagnosis, emotion_detection, etc)
        
        Returns:
            List of examples for that module
        """
        
        examples = [ex for ex in self.examples if ex.get("module") == module]
        logger.info(f"Examples for {module}: {len(examples)}")
        return examples
    
    
    def reset_counter(self) -> None:
        """
        Reset counter after fine-tuning is complete
        
        CALLED BY FINE-TUNING PIPELINE AFTER SUCCESS
        
        Moves old examples to archive and starts fresh accumulation
        
        Example:
            # After fine-tuning completes:
            counter.reset_counter()
            # Archived to: data/training/archived_2024-01-15.jsonl
            # Examples reset to 0
        """
        
        try:
            # Create archive filename with current date
            archive_file = self.accumulated_file.parent / f"archived_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
            
            # Move current file to archive
            if self.accumulated_file.exists():
                self.accumulated_file.rename(archive_file)
                logger.info(f"Archived examples to {archive_file}")
            
            # Clear in-memory list
            self.examples = []
            logger.info("Counter reset - ready for next fine-tuning cycle")
        
        except Exception as e:
            logger.error(f"Failed to reset counter: {str(e)}")
    
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about accumulated examples
        
        Returns:
            Detailed statistics for reporting/monitoring
        
        Example:
            stats = counter.get_statistics()
            print(f"Average rating: {stats['average_user_rating']:.1f}/5")
        """
        
        if not self.examples:
            return {
                "total": 0,
                "message": "No examples accumulated yet"
            }
        
        ratings = [ex["user_rating"] for ex in self.examples]
        
        return {
            "total_examples": len(self.examples),
            "threshold": self.threshold,
            "percentage_to_threshold": (len(self.examples) / self.threshold) * 100,
            
            "average_user_rating": statistics.mean(ratings),
            "median_user_rating": statistics.median(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            
            "quality_breakdown": self.count_by_quality(),
            "module_breakdown": self.count_by_module(),
            "user_breakdown": self.count_by_user(),
            
            "first_example_date": self.examples[0]["timestamp"] if self.examples else None,
            "last_example_date": self.examples[-1]["timestamp"] if self.examples else None,
            "ready_for_fine_tuning": self.is_ready_for_fine_tuning()["ready"]
        }
    
    
    def export_to_csv(self, output_file: str = "data/training/examples_export.csv") -> None:
        """Export examples to CSV for analysis"""
        
        import csv
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "module", "quality", "user_rating",
                    "user_id", "pet_id", "user_query", "ai_response"
                ])
                writer.writeheader()
                
                for example in self.examples:
                    writer.writerow({
                        "timestamp": example.get("timestamp"),
                        "module": example.get("module"),
                        "quality": example.get("quality"),
                        "user_rating": example.get("user_rating"),
                        "user_id": example.get("user_id"),
                        "pet_id": example.get("pet_id"),
                        "user_query": example.get("user_query", "")[:100],
                        "ai_response": example.get("ai_response", "")[:100]
                    })
            
            logger.info(f"Examples exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Failed to export examples: {str(e)}")

