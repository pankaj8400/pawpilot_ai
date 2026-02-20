"""
PawPilot AI Prompt Engineering Module

This module handles all prompt building, optimization, and management
for the PawPilot multimodal pet intelligence system.

Exports:
    - PromptBuilder: Main class for building PawPilot prompts
    - PromptEvaluator: Evaluate prompt quality and accuracy
    - PromptOptimizer: A/B test and optimize prompts
    - FewShotBuilder: Manage few-shot examples
    - PromptCache: Cache prompt results
"""
import logging
from src.rag.rag_pipline import RAGPipeline
from typing import Dict
from datetime import datetime
from src.utils.exceptions import CustomException
from datetime import datetime
import json
from typing import Dict, List 
from datetime import datetime
from pathlib import Path
from src.prompt_engineering.prompts import PawPilotPromptBuilder
from src.prompt_engineering.few_shot_builder import FewShotBuilder
from src.prompt_engineering.cache_manager import PromptCache
from src.utils.exceptions import CustomException
logger = logging.getLogger(__name__)

from .prompts import PawPilotPromptBuilder
from .prompt_evaluator import PromptEvaluator
from .prompt_optimizer import PromptOptimizer
from .few_shot_builder import FewShotBuilder
from .cache_manager import PromptCache

__all__ = [
    "PawPilotPromptBuilder",
    "PromptEvaluator",
    "PromptOptimizer",
    "FewShotBuilder",
    "PromptCache",
]


class Node4PromptEngineering:
    """
    NODE 4: Prompt Engineering for PawPilot AI
    
    Responsibilities:
    - Detect PawPilot module (skin diagnosis, emotion, emergency, product safety, behavior)
    - Load appropriate system prompt template
    - Retrieve relevant RAG context
    - Select relevant few-shot examples
    - Build optimized, module-specific prompt
    - Cache results for cost optimization
    """
    
    def __init__(self):
        """Initialize prompt engineering components"""
        self.prompt_builder = PawPilotPromptBuilder()
        self.few_shot_builder = FewShotBuilder(
            "src/prompt_engineering/templates/few_shot_examples.json"
        )
        self.prompt_cache = PromptCache()
        self.rag_pipeline = RAGPipeline()
        self.templates_dir = Path("src/prompt_engineering/templates")
    
    def engineer_prompt(self, state) -> Dict:
        """
        NODE 4: Build optimized prompt for PawPilot
        
        Args:
            state: WorkflowState object containing:
                - query: User's question
                - use_rag: Whether to use RAG context
                - context: Retrieved RAG documents
                - pet_profile: Pet information
                - module_type: Which PawPilot module to use
        
        Updates state with:
            - final_prompt: Complete, ready-to-use prompt
            - prompt_module: Which module was used
            - cache_hit: Whether this was cached
        """
        
        logger.info("=" * 60)
        logger.info("NODE 4: PROMPT ENGINEERING")
        logger.info("=" * 60)
        
        try:
            # ============================================================
            # STEP 1: DETECT MODULE TYPE
            # ============================================================
            logger.info("STEP 1: Detecting PawPilot module type...")
            
            module_type = self._detect_pawpilot_module(
                query=state.query,
                input_type=state.get("input_type", "text")
            )
            
            state["prompt_module"] = module_type
            logger.info(f"✓ Module detected: {module_type}")
            
            
            # ============================================================
            # STEP 2: CHECK CACHE (Optimization)
            # ============================================================
            logger.info("STEP 2: Checking prompt cache...")
            
            cache_key = self._generate_cache_key(state)
            cached_prompt = self.prompt_cache.get(
                prompt=state.query,
                model=state.get("model_to_use", "gpt-4-turbo"),
                pet_id=state.get("pet_id")
            )
            
            if cached_prompt:
                logger.info("✓ Cache HIT - Using cached prompt")
                state["final_prompt"] = cached_prompt
                state["cache_hit"] = True
                return state
            
            state["cache_hit"] = False
            logger.info("✓ Cache MISS - Building new prompt")
            
            
            # ============================================================
            # STEP 3: LOAD SYSTEM PROMPT TEMPLATE
            # ============================================================
            logger.info("STEP 3: Loading system prompt template...")
            
            system_prompt = self._load_system_prompt(module_type)
            logger.info(f"✓ System prompt loaded for module: {module_type}")
            
            
            # ============================================================
            # STEP 4: SELECT PROMPT TEMPLATE (RAG vs Standard)
            # ============================================================
            logger.info("STEP 4: Selecting prompt template...")
            
            if state.get("use_rag"):
                state["prompt_template_type"] = "rag_enhanced"
                logger.info("✓ Using RAG-enhanced template")
            else:
                state["prompt_template_type"] = "standard"
                logger.info("✓ Using standard template")
            
            
            # ============================================================
            # STEP 5: RETRIEVE RAG CONTEXT (if needed)
            # ============================================================
            logger.info("STEP 5: Retrieving RAG context...")
            
            rag_context = ""
            if state.get("use_rag"):
                rag_context = self._retrieve_rag_context(
                    query=state.query,
                    module_type=module_type,
                    pet_profile=state.get("pet_profile", {})
                )
                logger.info(f"✓ RAG context retrieved ({len(rag_context)} chars)")
            else:
                logger.info("✓ RAG context skipped (not needed for this query)")
            
            state["rag_context_used"] = rag_context[:100] + "..." if rag_context else "None"
            
            
            # ============================================================
            # STEP 6: SELECT RELEVANT FEW-SHOT EXAMPLES
            # ============================================================
            logger.info("STEP 6: Selecting few-shot examples...")
            
            few_shot_examples = self._select_few_shot_examples(
                query=state.query,
                module_type=module_type,
                num_examples=2  # Default: 2 examples
            )
            
            few_shot_formatted = self._format_few_shot_examples(few_shot_examples)
            logger.info(f"✓ Selected {len(few_shot_examples)} few-shot examples")
            
            
            # ============================================================
            # STEP 7: BUILD FINAL PROMPT (Module-Specific)
            # ============================================================
            logger.info("STEP 7: Building final prompt for module...")
            
            final_prompt = self._build_module_specific_prompt(
                module_type=module_type,
                system_prompt=system_prompt,
                user_query=state.query,
                context=state.get("context", ""),
                rag_context=rag_context,
                few_shot_examples=few_shot_formatted,
                pet_profile=state.get("pet_profile", {}),
                additional_data=state.get("additional_data", {})
            )
            
            state["final_prompt"] = final_prompt
            logger.info(f"✓ Final prompt built ({len(final_prompt)} chars)")
            
            
            # ============================================================
            # STEP 8: CACHE THE PROMPT (for future use)
            # ============================================================
            logger.info("STEP 8: Caching prompt for future use...")
            
            self.prompt_cache.set(
                prompt=state.query,
                model=state.get("model_to_use", "gpt-4-turbo"),
                response=final_prompt,
                pet_id=state.get("pet_id")
            )
            logger.info("✓ Prompt cached successfully")
            
            
            # ============================================================
            # STEP 9: ADD METADATA TO STATE
            # ============================================================
            logger.info("STEP 9: Adding metadata...")
            
            state["prompt_engineering_metadata"] = {
                "module": module_type,
                "template_type": state["prompt_template_type"],
                "has_rag_context": state.get("use_rag", False),
                "few_shot_count": len(few_shot_examples),
                "prompt_length": len(final_prompt),
                "cache_hit": state["cache_hit"],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✓ Metadata added")
            
            logger.info("=" * 60)
            logger.info("NODE 4 COMPLETE - Prompt ready for inference")
            logger.info("=" * 60)
            
            
        
        except Exception as e:
            logger.error(f"ERROR in Node 4: {str(e)}", exc_info=True)
            state.get("errors", []).append(f"Prompt Engineering Error: {str(e)}")
            raise CustomException(f"Node 4 Prompt Engineering failed: {str(e)}", str(e))
        
        finally:
            return state
    
    # ====================================================================
    # HELPER METHODS
    # ====================================================================
    
    def _detect_pawpilot_module(self, query: str, input_type: str) -> str:
        """
        Detect which PawPilot module should handle this query
        
        Returns: "skin_diagnosis" | "emotion_detection" | "emergency" | 
                 "product_safety" | "behavior" | "general_qa"
        """
        
        query_lower = query.lower()
        
        # PRIORITY 1: EMERGENCY (highest priority)
        emergency_keywords = [
            "emergency", "urgent", "critical", "choking", "seizure", 
            "poisoning", "bleeding", "can't breathe", "collapse",
            "dying", "help now", "immediate", "911", "call vet"
        ]
        if any(kw in query_lower for kw in emergency_keywords):
            logger.debug("Module detected: EMERGENCY (priority 1)")
            return "emergency"
        
        # PRIORITY 2: EMOTION/AUDIO (if audio input)
        if input_type in ["audio", "voice"]:
            logger.debug("Module detected: EMOTION_DETECTION (audio input)")
            return "emotion_detection"
        
        # PRIORITY 3: SKIN/HEALTH DIAGNOSIS (if image input)
        if input_type in ["image", "photo", "visual"]:
            skin_keywords = [
                "rash", "skin", "wound", "sore", "infection", "itching",
                "discharge", "hot spot", "scab", "bump", "lesion", "paw",
                "ear", "eye", "fur", "coat"
            ]
            if any(kw in query_lower for kw in skin_keywords):
                logger.debug("Module detected: SKIN_DIAGNOSIS (image + keywords)")
                return "skin_diagnosis"
        
        # PRIORITY 4: PRODUCT SAFETY
        product_keywords = [
            "food", "treat", "ingredient", "safe", "toxic", "product",
            "toy", "supplement", "shampoo", "poison", "allergen"
        ]
        if any(kw in query_lower for kw in product_keywords):
            logger.debug("Module detected: PRODUCT_SAFETY")
            return "product_safety"
        
        # PRIORITY 5: EMOTION/BEHAVIOR (text queries)
        emotion_keywords = [
            "sound", "bark", "meow", "cry", "whine", "growl", "emotion",
            "feeling", "scared", "anxious", "happy", "aggressive", "stressed",
            "training", "behavior", "biting", "jumping", "pulling"
        ]
        if any(kw in query_lower for kw in emotion_keywords):
            logger.debug("Module detected: EMOTION_DETECTION or BEHAVIOR")
            if any(bk in query_lower for bk in ["training", "behavior", "teach", "command"]):
                return "behavior"
            return "emotion_detection"
        
        # PRIORITY 6: HEALTH/DIAGNOSIS
        health_keywords = [
            "sick", "illness", "disease", "symptom", "health", "vet",
            "doctor", "condition", "problem", "issue", "concern"
        ]
        if any(kw in query_lower for kw in health_keywords):
            logger.debug("Module detected: SKIN_DIAGNOSIS (health keywords)")
            return "skin_diagnosis"
        
        # DEFAULT: General QA
        logger.debug("Module detected: GENERAL_QA (default)")
        return "general_qa"
    
    
    def _load_system_prompt(self, module_type: str) -> str:
        """Load system prompt for specific module from JSON"""
        
        try:
            system_prompts_file = self.templates_dir / "system_prompts.json"
            
            with open(system_prompts_file, 'r') as f:
                all_prompts = json.load(f)
            
            # Map module_type to prompt key
            prompt_key_map = {
                "skin_diagnosis": "skin_health_diagnostic",
                "emotion_detection": "voice_emotion_translator",
                "emergency": "emergency_assistant",
                "product_safety": "product_safety_evaluator",
                "behavior": "behavior_training_coach",
                "general_qa": "general_assistant"
            }
            
            prompt_key = prompt_key_map.get(module_type, "general_assistant")
            system_prompt = all_prompts.get(prompt_key, {})
            
            # Build system prompt string
            system_text = f"""You are: {system_prompt.get('role', 'a helpful assistant')}

            Context: {system_prompt.get('context', '')}
            
            Key Principles:
            {chr(10).join(f"- {p}" for p in system_prompt.get('key_principles', []))}
            
            Tone: {system_prompt.get('tone', 'Professional and helpful')}
            
            Constraints:
            {chr(10).join(f"- {c}" for c in system_prompt.get('constraints', []))}
            """
            
            return system_text
        
        except Exception as e:
            logger.warning(f"Could not load system prompt for {module_type}: {str(e)}")
            return "You are a helpful AI assistant for pet care."
    
    
    def _retrieve_rag_context(self, query: str, module_type: str, pet_profile: Dict) -> str:
        """Retrieve relevant context from RAG pipeline"""
        
        try:
            # Determine which RAG database to search based on module
            rag_context_map = {
                "skin_diagnosis": "skin_conditions",
                "emotion_detection": "emotion_detection",
                "emergency": "emergency_protocols",
                "product_safety": "product_safety",
                "behavior": "behavior_training",
                "general_qa": "general_knowledge"
            }
            
            context_type = rag_context_map.get(module_type, "general_knowledge")
            
            # Retrieve from RAG
            rag_results = self.rag_pipeline.retriever(
                query=query,
                top_k=5
            )
            
            # Format RAG results
            rag_context = "## KNOWLEDGE BASE REFERENCE\n\n"
            for i, result in enumerate(rag_results, 1):
                rag_context += f"[{i}] {result.get('content', '')}\n"
                if i >= 3:  # Limit to top 3 results
                    break
            
            return rag_context
        
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {str(e)}")
            return ""
    
    
    def _select_few_shot_examples(self, query: str, module_type: str, num_examples: int = 2) -> List[Dict]:
        """Select most relevant few-shot examples based on query"""
        
        try:
            # Try to get examples similar to query
            examples = self.few_shot_builder.select_relevant_examples(
                query=query,
                module_type=module_type,
                num_examples=num_examples
            )
            
            # If not enough, get any high-confidence examples
            if len(examples) < num_examples:
                high_conf_examples = self.few_shot_builder.get_high_confidence_examples(
                    module_type=module_type,
                    min_confidence=0.8
                )
                examples.extend(high_conf_examples[:num_examples - len(examples)])
            
            return examples[:num_examples]
        
        except Exception as e:
            logger.warning(f"Few-shot selection failed: {str(e)}")
            return []
    
    
    def _format_few_shot_examples(self, examples: List[Dict]) -> str:
        """Format few-shot examples into prompt-ready string"""
        
        if not examples:
            return ""
        
        formatted = "\n## FEW-SHOT EXAMPLES\n\n"
        
        for i, example in enumerate(examples, 1):
            formatted += f"### Example {i}:\n"
            
            if "input" in example:
                formatted += f"**Input:** {example['input']}\n\n"
            elif "q" in example:
                formatted += f"**Question:** {example['q']}\n\n"
            
            if "output" in example:
                formatted += f"**Output:**\n{example['output']}\n\n"
            elif "a" in example:
                formatted += f"**Answer:** {example['a']}\n\n"
        
        return formatted
    
    
    def _build_module_specific_prompt(
        self,
        module_type: str,
        system_prompt: str,
        user_query: str,
        context: str,
        rag_context: str,
        few_shot_examples: str,
        pet_profile: Dict,
        additional_data: Dict
    ) -> str:
        """Build module-specific final prompt"""
        
        # Build pet profile section
        pet_profile_section = ""
        if pet_profile:
            pet_profile_section = f"""
            ## PET INFORMATION
            - Name: {pet_profile.get('name', 'Unknown')}
            - Species: {pet_profile.get('species', 'Dog')}
            - Breed: {pet_profile.get('breed', 'Unknown')}
            - Age: {pet_profile.get('age', 'Unknown')} years
            - Weight: {pet_profile.get('weight', 'Unknown')} kg
            - Known Allergies: {', '.join(pet_profile.get('allergies', ['None']))}
            - Medical History: {pet_profile.get('medical_history', 'None reported')}
            """
        
        # Module-specific prompt structures
        if module_type == "emergency":
            # Emergency: CRITICAL FORMAT - numbered steps, no context
            final_prompt = f"""{system_prompt}
            
            {pet_profile_section}
            
            🚨 EMERGENCY SITUATION:
            {user_query}
            
            {rag_context}
            
            RESPOND WITH:
            1. Severity assessment (LIFE-THREATENING / SERIOUS / URGENT)
            2. Critical time window
            3. Numbered immediate actions (1, 2, 3...)
            4. What NOT to do (use ❌)
            5. Vet urgency level
            6. Equipment needed
            
            CRITICAL: Use NUMBERED STEPS ONLY. No paragraphs. Be directive and clear."""

        elif module_type == "skin_diagnosis":
            # Skin diagnosis: Detail-oriented with RAG
            final_prompt = f"""{system_prompt}
                    
                    {pet_profile_section}
                    
                    KNOWLEDGE BASE:
                    {rag_context}
                    
                    {few_shot_examples}
                    
                    ## ANALYSIS TASK
                    Analyze the following symptom/image description:
                    
                    {user_query}
                    
                    Additional context:
                    {context}
                    
                    PROVIDE:
                    1. Observations (what you see/identify)
                    2. Possible conditions with likelihood levels
                    3. Severity: Low/Medium/High/Emergency
                    4. Urgency: When should they see a vet?
                    5. First aid steps (numbered)
                    6. What to monitor
                    7. Vet visit recommendation"""

        elif module_type == "emotion_detection":
                # Emotion detection: Uses EmoDetect framework
                final_prompt = f"""{system_prompt}
                
                {pet_profile_section}
                
                ## EMOTION DETECTION FRAMEWORK (EmoDetect)
                {rag_context}
                
                {few_shot_examples}
                
                ## INPUT TO ANALYZE
                {user_query}
                
                Additional information:
                {context}
                
                PROVIDE:
                1. Primary emotion detected
                2. Confidence level (High/Medium/Low)
                3. Key body language indicators identified
                4. Root cause/trigger analysis
                5. Recommended actions to help the pet
                6. When to consult a vet"""

        elif module_type == "product_safety":
            # Product safety: Structured evaluation
            final_prompt = f"""{system_prompt}
                    
                    {pet_profile_section}
                    
                    SAFETY DATABASE:
                    {rag_context}
                    
                    {few_shot_examples}
                    
                    ## PRODUCT TO EVALUATE
                    {user_query}
                    
                    Product details:
                    {context}
                    
                    PROVIDE:
                    1. Product name and type
                    2. Ingredients breakdown
                    3. Safety assessment (Safe/Caution/Not Safe)
                    4. 🚨 Flag any toxic ingredients
                    5. Allergen risks for this pet
                    6. Safety score (1-10)
                    7. Portion size recommendation
                    8. Better alternatives
                    9. Final recommendation (YES/NO)"""

        elif module_type == "behavior":
            # Behavior/training: Practical step-by-step
            final_prompt = f"""{system_prompt}
                    
                    {pet_profile_section}
                    
                    {few_shot_examples}
                    
                    ## BEHAVIOR CHALLENGE
                    {user_query}
                    
                    Context:
                    {context}
                    
                    PROVIDE:
                    1. Understanding the behavior (why it happens)
                    2. Root cause analysis
                    3. Step-by-step training plan (phases)
                    4. Timeline expectations (weeks/months)
                    5. Success indicators
                    6. Common mistakes to avoid
                    7. When to seek professional help
                    
                    Use encouraging language and emphasize consistency."""

        else:  # general_qa
            # General Q&A: Standard format
            final_prompt = f"""{system_prompt}

        {pet_profile_section}
        
        {rag_context}
        
        {few_shot_examples}
        
        ## QUESTION
        {user_query}
        
        Context:
        {context}
        
        Please provide a helpful, clear answer based on pet care expertise."""

        return final_prompt
    
    
    def _generate_cache_key(self, state: Dict) -> str:
        """Generate cache key for this request"""
        return f"{state.get('pet_id', 'general')}:{state.get('query', '')[:50]}"
