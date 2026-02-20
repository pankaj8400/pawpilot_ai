from typing import Dict, Optional, List
import json
from pathlib import Path
import logging
from .food_model_prompts import (
    get_response_prompt as get_food_response_prompt,
    route_food_query
)
from .Injury_model_prompt import get_prompt_for_context, get_generation_prompt_for_context, detect_injury_context_from_text  
logger = logging.getLogger(__name__)

class PawPilotPromptBuilder:
    """Build specialized prompts for PawPilot AI modules"""
    
    def __init__(self):
        self.templates_dir = Path("AI_Model/src/prompt_engineering/templates")
        self.load_all_templates()
    
    def load_all_templates(self):
        """Load all PawPilot-specific templates"""
        self.system_prompts = self._load_json("system_prompts.json")
        self.vision_prompts = self._load_json("vision_prompts.json")
        self.audio_prompts = self._load_json("audio_prompt.json")  # Note: file is audio_prompt.json not audio_prompts.json
        self.few_shot = self._load_json("few_shot_examples.json")
        self.rag_templates = self._load_json("rag_context_templates.json")
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file with error handling for missing files"""
        path = self.templates_dir / filename
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Template file not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return {}
                
    # ==========================================
    # EMOTION DETECTION PROMPT
    # ==========================================
    
    def build_emotion_detection_prompt(
        self,
        image_features: str,
        audio_analysis: str,
        pet_profile: Dict,
        rag_emotion_data: str
    ) -> str:
        """Build prompt for EmoDetect emotion analysis"""
        
        system = self.system_prompts["emotion-detection"]
        
        prompt = f"""You are an {system['role']}
        
        {system['context']}
        
        EMOTION DETECTION FRAMEWORK (From EmoDetect RAG):
        {rag_emotion_data}
        
        PET CONTEXT:
        - Name: {pet_profile.get('name')}
        - Breed: {pet_profile.get('breed')}
        - Age: {pet_profile.get('age')} years
        - Personality: {pet_profile.get('personality', 'Unknown')}
        - Recent Events: {pet_profile.get('recent_events', 'None reported')}
        
        VISUAL ANALYSIS:
        {image_features}
        
        AUDIO ANALYSIS:
        {audio_analysis}
        
        DETECTION TASK:
        1. Identify body language indicators (using EmoDetect framework)
        2. Analyze audio patterns and vocalizations
        3. Consider pet's personality and recent context
        4. Cross-reference with EmoDetect emotion taxonomy
        5. Assess confidence level of emotion detection
        6. Recommend appropriate actions
        
        OUTPUT FORMAT:
        ## Primary Emotion
        [Emotion from EmoDetect list]
        
        ## Confidence Level
        High / Medium / Low
        
        ## Key Indicators Observed
        - Body Language: [indicators]
        - Vocalizations: [audio patterns]
        - Context Clues: [situational factors]
        
        ## Root Cause Analysis
        [What triggered this emotion]
        
        ## Recommended Actions
        1. [What to do]
        2. [How to help]
        3. [When to seek help]
        
        ## Important Notes
        Be specific about which body parts indicate which emotions using EmoDetect framework."""
        
        return prompt
    
    # ==========================================
    # EMERGENCY RESPONSE PROMPT
    # ==========================================
    
    def build_emergency_prompt(
        self,
        emergency_type: str,
        symptoms: str,
        pet_profile: Dict,
        rag_emergency_protocols: str
    ) -> str:
        """Build CRITICAL CARE prompt for emergencies"""
        
        system = self.system_prompts["injury-assistance"]
        
        prompt = f"""You are a {system['role']}

        {system['context']}
        
        KEY PRINCIPLES FOR LIFE-SAVING RESPONSES:
        {chr(10).join(f"- {p}" for p in system['key_principles'])}
        
        PET INFORMATION:
        - Age: {pet_profile.get('age')} years
        - Weight: {pet_profile.get('weight')} kg
        - Medical Conditions: {pet_profile.get('medical_conditions', 'None reported')}
        - Current Medications: {pet_profile.get('medications', 'None')}
        
        EMERGENCY TYPE: {emergency_type}
        
        SYMPTOMS REPORTED:
        {symptoms}
        
        EMERGENCY PROTOCOLS (From RAG):
        {rag_emergency_protocols}
        
        CRITICAL RESPONSE REQUIRED:
        ⚠️ PROVIDE NUMBERED STEPS ONLY - NO PARAGRAPHS
        ⚠️ START WITH SEVERITY ASSESSMENT
        ⚠️ INCLUDE WHAT NOT TO DO
        ⚠️ SPECIFY TIME WINDOWS
        
        OUTPUT FORMAT (STRICT):
        ## 🚨 SEVERITY LEVEL
        [LIFE-THREATENING / SERIOUS / URGENT]
        
        ## ⏱️ TIME CRITICAL WINDOW
        [How much time available before escalation needed]
        
        ## 🔴 IMMEDIATE ACTIONS (IN ORDER):
        1. [Action - be specific]
        2. [Action - be specific]
        3. [Action - be specific]
        4. [Action - be specific]
        
        ## ❌ WHAT NOT TO DO:
        - Do NOT [something dangerous]
        - Do NOT [something dangerous]
        - Do NOT [something dangerous]
        
        ## 📞 VET URGENCY
        [CALL IMMEDIATELY / Go to ER now / Vet within 1 hour]
        
        ## 🩺 EQUIPMENT NEEDED
        [List specific items needed]
        
        ## 👁️ WARNING SIGNS FOR ESCALATION
        [When to stop and go to vet immediately]"""
                
        return prompt
    
    # ==========================================
    # PRODUCT SAFETY PROMPT
    # ==========================================
    
    def build_product_analysis_prompt(
        self,
        product_info: Dict,
        pet_profile: Dict,
        rag_safety_database: str
    ) -> str:
        """Build prompt for product safety evaluation"""
        
        system = self.system_prompts["packaged-product-scanner"]
        
        prompt = f"""You are a {system['role']}
        
        {system['context']}
        
        SAFETY DATABASE REFERENCE:
        {rag_safety_database}
        
        PET PROFILE:
        - Species: {pet_profile.get('species')}
        - Age: {pet_profile.get('age')} years
        - Weight: {pet_profile.get('weight')} kg
        - Known Allergies: {', '.join(pet_profile.get('allergies', ['None']))}
        - Health Conditions: {pet_profile.get('health_conditions', 'None reported')}
        
        PRODUCT TO EVALUATE:
        - Name: {product_info.get('name')}
        - Type: {product_info.get('type')} (food/treat/toy/supplement)
        - Ingredients: {product_info.get('ingredients')}
        - Price: {product_info.get('price', 'Unknown')}
        
        EVALUATION STEPS:
        1. Check each ingredient against safety database
        2. Flag any toxic substances
        3. Assess portion/size appropriateness for pet
        4. Check for allergen risks specific to this pet
        5. Evaluate nutritional content
        6. Compare price vs value
        7. Suggest alternatives if needed
        
        OUTPUT FORMAT:
        ## 🏷️ Product Name & Type
        [Info]
        
        ## ✅ Safety Assessment
        Safe / Caution / Not Safe
        
        ## 🚨 Toxic Ingredients Found
        [List any toxic ingredients, or "None found"]
        
        ## ⚠️ Allergen Concerns
        [Any concerns for this specific pet]
        
        ## 📊 Safety Score
        [1-10 scale with explanation]
        
        ## 💡 Better Alternatives
        [Healthier/safer options]
        
        ## 📏 Appropriate Portion Size
        [Specific amount for this pet's weight]
        
        ## 💰 Value Assessment
        [Is it worth the price?]
        
        ## 🎯 Final Recommendation
        [Clear recommendation for this pet]"""
        
        return prompt
    
    # ==========================================
    # VISION - TOY CLASSIFIER PROMPT
    # ==========================================
    
    def build_toy_classifier_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for toy classification and safety analysis
        
        Args:
            predicted_class: The classified toy type from vision model
            confidence_score: Model confidence (0-1)
            user_query: User's question about the toy
            rag_context: Retrieved toy information from RAG
            pet_profile: Optional pet info for personalized recommendations
        """
        
        vision_config = self.vision_prompts.get("toy-safety-detection", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        thresholds = self.vision_prompts.get("confidence_thresholds", {})
        
        # Determine confidence level
        if confidence_score >= thresholds.get("high_confidence", 0.85):
            confidence_level = "High"
        elif confidence_score >= thresholds.get("medium_confidence", 0.65):
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Build pet context if available
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Breed: {pet_profile.get('breed', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Size: {pet_profile.get('size', 'Unknown')}
- Chewing Strength: {pet_profile.get('chewing_strength', 'Unknown')}
"""
        
        prompt = f"""You are a {system.get('role', 'pet product advisor')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Friendly and helpful')}

IMAGE ANALYSIS RESULT:
- Identified Product: {predicted_class}
- Confidence: {confidence_score:.1%} ({confidence_level})
{pet_context}
KNOWLEDGE BASE REFERENCE:
{rag_context if rag_context else "No additional product information available."}

USER QUESTION:
{user_query}

OUTPUT FORMAT:
## 🧸 Product Identification
[Identified toy type and confidence]

## ✅ Safety Assessment
[Safety evaluation for different pet types/sizes]

## 🐕 Best Suited For
[Which pets this toy is appropriate for]

## 🎮 Play Tips
[How to use this toy effectively]

## 🧹 Care & Maintenance
[How to clean and when to replace]

## ⚠️ Supervision Recommendation
[Always end with supervision advice]

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 250)} words
- Be specific about safety concerns
- Include size/breed recommendations"""
        
        return prompt
    
    # ==========================================
    # VISION - DISEASES CLASSIFIER PROMPT
    # ==========================================
    
    def build_diseases_classifier_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for pet skin condition/disease analysis
        
        Args:
            predicted_class: The classified condition from vision model
            confidence_score: Model confidence (0-1)
            user_query: User's question about the condition
            rag_context: Retrieved medical information from RAG
            pet_profile: Optional pet info for personalized assessment
        """
        
        vision_config = self.vision_prompts.get("skin-and-health-diagnostic", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        thresholds = self.vision_prompts.get("confidence_thresholds", {})
        emergency_triggers = vision_config.get("emergency_triggers", [])
        
        # Determine confidence level
        if confidence_score >= thresholds.get("high_confidence", 0.85):
            confidence_level = "High"
        elif confidence_score >= thresholds.get("medium_confidence", 0.65):
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Check for emergency keywords in query
        is_emergency = any(trigger in user_query.lower() for trigger in emergency_triggers)
        
        # Build pet context if available
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Breed: {pet_profile.get('breed', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Weight: {pet_profile.get('weight', 'Unknown')} kg
- Known Allergies: {', '.join(pet_profile.get('allergies', ['None known']))}
- Medical History: {pet_profile.get('medical_history', 'None reported')}
"""
        
        emergency_notice = ""
        if is_emergency:
            emergency_notice = """
🚨 EMERGENCY INDICATORS DETECTED - PRIORITIZE IMMEDIATE GUIDANCE 🚨
"""
        
        prompt = f"""You are a {system.get('role', 'veterinary AI assistant')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Professional and caring')}
{emergency_notice}
IMAGE ANALYSIS RESULT:
- Possible Condition Detected: {predicted_class}
- Model Confidence: {confidence_score:.1%} ({confidence_level})
{pet_context}
MEDICAL KNOWLEDGE BASE REFERENCE:
{rag_context if rag_context else "No additional medical information available from knowledge base."}

USER CONCERN:
{user_query}

OUTPUT FORMAT:
## 👁️ Visual Observation
[What the image analysis detected]

## 🔍 Possible Condition
[Condition name with confidence - NEVER state as definitive diagnosis]

## ⚠️ Severity Level
[{' / '.join(template.get('severity_levels', ['Low', 'Medium', 'High', 'Emergency']))}]

## 🩹 Immediate Care Steps
1. [First aid action]
2. [Comfort measure]
3. [What to monitor]

## 🏥 Veterinary Recommendation
[When to see vet: Immediately / Within 24 hours / Within 48-72 hours / Within 1 week]

## ⚕️ Disclaimer
*This is an AI-assisted assessment and NOT a medical diagnosis. Always consult a licensed veterinarian for proper examination and treatment.*

CONSTRAINTS:
- Minimum {template.get('constraints', {}).get('max_words', 300)} words
- Always phrase as "possible condition" not "diagnosis"
- Always recommend veterinary consultation
- Include severity assessment
- Be empathetic but honest"""
        
        return prompt
    
    # ==========================================
    # VISION - DEFAULT/GENERAL PROMPT
    # ==========================================
    
    def build_vision_default_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str = "",
        strategy : str = "default"
    ) -> str:
        """
        Build a general vision analysis prompt for unclassified images
        """
        # Default configuration
        system = {
            'role': 'pet care assistant',
            'context': 'You are helping users understand and care for their pets.',
            'key_principles': ['Be helpful', 'Be safe', 'Be accurate'],
            'tone': 'Friendly and informative'
        }
        
        if strategy == "skin-and-health-diagnostic":
            vision_config = self.vision_prompts.get("skin-and-health-diagnostic", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "toy-safety-detection":
            vision_config = self.vision_prompts.get("toy-safety-detection", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "emotion-detection":
            vision_config = self.vision_prompts.get("emotion-detection", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "injury-assitance":
            vision_config = self.vision_prompts.get("injury-assistance", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "packaged-product-scanner":
            vision_config = self.vision_prompts.get("packaged-product-scanner", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "pet-food-image-analysis":
            vision_config = self.vision_prompts.get("pet-food-image-analysis", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "full-body-scan":
            vision_config = self.vision_prompts.get("full-body-scan", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "parasite-detection":
            vision_config = self.vision_prompts.get("parasite-detection", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "poop-vomit-detection":
            vision_config = self.vision_prompts.get("poop-vomit-detection", {})
            system = vision_config.get("system_prompt", {})
        elif strategy == "home-environment-safety-scan":
            vision_config = self.vision_prompts.get("home-environment-safety-scan", {})
            system = vision_config.get("system_prompt", {})
        
        prompt = f"""You are a {system.get('role', 'pet care assistant')}.
        
        # CONTEXT
        {system.get('context', '')}
        
        # KEY PRINCIPLES
        {chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

        # TONE
        {system.get('tone', '')}
        
        """
        return prompt
    
    # ==========================================
    # INJURY ASSISTANCE PROMPT
    # ==========================================
    def build_injury_assistance_prompt(self, user_query:str, rag_context: str="" ) -> str:
        """
        Build prompt for Injury analysis
        
        Args:
            injury_type: Type of the injure- cut, bite, scratch, etc
            injured_type : human, dog, cat
            severity_type: normal, moderate, high
            user_query: User's question about the condition
            rag_context: Retrieved medical information from RAG
            pet_profile: Optional pet info for personalized assessment
        """
        injury_context = detect_injury_context_from_text(user_query) #get the context about the injury that what is the injury actually
        specific_prompt = get_generation_prompt_for_context(injury_context) #generate the prompt for the particular injury

        final_prompt = specific_prompt + rag_context + user_query # make the final prompt combining all the text 
        return final_prompt
    
    # ==========================================
    # PARASITE DETECTION PROMPT
    # ==========================================
    def build_parasite_detection_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for parasitic infection detection and analysis
        """
        vision_config = self.vision_prompts.get("parasite-detection", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        thresholds = self.vision_prompts.get("confidence_thresholds", {})
        
        if confidence_score >= thresholds.get("high_confidence", 0.85):
            confidence_level = "High"
        elif confidence_score >= thresholds.get("medium_confidence", 0.65):
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Breed: {pet_profile.get('breed', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Weight: {pet_profile.get('weight', 'Unknown')} kg
- Current Treatments: {pet_profile.get('treatments', 'None')}
"""
        
        prompt = f"""You are a {system.get('role', 'veterinary parasitologist')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Professional, urgent but calm')}

PARASITE ANALYSIS:
- Suspected Parasite: {predicted_class}
- Model Confidence: {confidence_score:.1%} ({confidence_level})
{pet_context}
PARASITOLOGY KNOWLEDGE BASE:
{rag_context if rag_context else "No additional parasite information available."}

USER CONCERN:
{user_query}

OUTPUT FORMAT:
## 🪱 Parasite Identified
[Suspected parasite type and confidence]

## 🚨 Severity Assessment
[Mild / Moderate / Severe / Critical]

## ⚠️ Health Risks
[Specific health risks for this pet]

## 🏥 Immediate Actions
1. [First action]
2. [Second action]
3. [Protective measure]

## 📞 Veterinary Care Required
[IMMEDIATE / Within 24 hours / Within 48 hours]

## 🛡️ Prevention & Treatment
[Treatment options and prevention strategies]

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 300)} words
- Always recommend veterinary consultation immediately
- Include transmission risks to other pets"""
        
        return prompt
    
    # ==========================================
    # POOP/VOMIT DETECTION PROMPT
    # ==========================================
    def build_poop_vomit_detection_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for gastrointestinal health assessment
        """
        vision_config = self.vision_prompts.get("poop-vomit-detection", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        thresholds = self.vision_prompts.get("confidence_thresholds", {})
        
        if confidence_score >= thresholds.get("high_confidence", 0.85):
            confidence_level = "High"
        elif confidence_score >= thresholds.get("medium_confidence", 0.65):
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Species: {pet_profile.get('breed', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Diet: {pet_profile.get('diet', 'Unknown')}
- Recent Diet Changes: {pet_profile.get('recent_diet_changes', 'None')}
- Medical History: {pet_profile.get('medical_history', 'None')}
"""
        
        prompt = f"""You are a {system.get('role', 'veterinary diagnostician')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Clinical, caring, actionable')}

GI HEALTH ANALYSIS:
- Finding Type: {predicted_class}
- Model Confidence: {confidence_score:.1%} ({confidence_level})
{pet_context}
GASTROINTESTINAL KNOWLEDGE BASE:
{rag_context if rag_context else "No additional GI information available."}

USER CONCERN:
{user_query}

OUTPUT FORMAT:
## 👀 Visual Assessment
[What the image shows - color, consistency, contents]

## 🔍 Possible Causes
[Likely reasons for this presentation]

## ⚠️ Severity Level
[{' / '.join(template.get('severity_levels', ['Low', 'Medium', 'High', 'Emergency']))}]

## 🏥 Immediate Care
1. [Hydration step]
2. [Diet modification]
3. [Monitoring protocol]

## 📞 When to See Vet
[Duration: Immediately / Within hours / Within 24 hours]

## 🚨 Emergency Symptoms to Watch
[When to escalate to emergency care]

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 280)} words
- Always recommend veterinary consultation
- Include hydration importance"""
        
        return prompt
    
    # ==========================================
    # HOME ENVIRONMENT SAFETY SCAN PROMPT
    # ==========================================
    def build_home_environment_safety_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for home environment hazard assessment
        """
        vision_config = self.vision_prompts.get("home-environment-safety-scan", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Age: {pet_profile.get('age', 'Unknown')} years
- Size: {pet_profile.get('size', 'Unknown')}
- Curiosity Level: {pet_profile.get('curiosity_level', 'Unknown')}
- Special Needs: {pet_profile.get('special_needs', 'None')}
"""
        
        prompt = f"""You are a {system.get('role', 'pet safety specialist')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Proactive, helpful, safety-focused')}

ENVIRONMENT ANALYSIS:
- Room/Area Type: {predicted_class}
{pet_context}
SAFETY KNOWLEDGE BASE:
{rag_context if rag_context else "No additional safety information available."}

USER QUESTION:
{user_query}

OUTPUT FORMAT:
## 🏠 Environment Assessment
[What type of space and overall safety impression]

## 🚨 Identified Hazards
1. [Hazard - be specific]
2. [Hazard - be specific]
3. [Hazard - be specific]

## ⚠️ Risk Level
[{' / '.join(template.get('risk_levels', ['Low', 'Medium', 'High']))}]

## ✅ Safety Improvements
1. [Remove/Secure this item]
2. [Move this out of reach]
3. [Add this safety feature]

## 📋 Best Practices
- [Practice for this environment]
- [Practice for pet safety]
- [Emergency preparation]

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 300)} words
- Be specific about hazards and solutions
- Consider different pet sizes and ages"""
        
        return prompt
    
    # ==========================================
    # PACKAGED PRODUCT SCANNER PROMPT
    # ==========================================
    def build_packaged_product_scanner_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for packaged product ingredient analysis and safety
        """
        vision_config = self.vision_prompts.get("packaged-product-scanner", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Species: {pet_profile.get('species', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Weight: {pet_profile.get('weight', 'Unknown')} kg
- Known Allergies: {', '.join(pet_profile.get('allergies', ['None']))}
- Health Conditions: {pet_profile.get('health_conditions', 'None')}
"""
        
        prompt = f"""You are a {system.get('role', 'pet product expert')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Informative, professional, consumer-friendly')}

PRODUCT ANALYSIS:
- Product Identified: {predicted_class}
- Analysis Confidence: {confidence_score:.1%}
{pet_context}
PRODUCT SAFETY DATABASE:
{rag_context if rag_context else "No additional product information available."}

USER QUERY:
{user_query}

OUTPUT FORMAT:
## 📦 Product Identification
[Product name, type, and brand]

## 📋 Ingredient Analysis
[Key ingredients and their quality assessment]

## 🥗 Nutritional Profile
[AAFCO compliance, nutritional completeness]

## ✅ Safety Assessment
[Ingredient safety, allergen concerns, harmful substances]

## 💬 Value Verdict 
[Price vs quality assessment and recommendations]

## 💡 Better Alternatives
[Similar products with better nutritional profiles]

CONSTRAINTS:
- Minimum {template.get('constraints', {}).get('min_words', 300)} words
- Extract and list key ingredients
- Flag any concerning additives or allergens
- Include AAFCO reference if available"""
        
        return prompt
    
    # ==========================================
    # FULL BODY SCAN PROMPT
    # ==========================================
    def build_full_body_scan_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for comprehensive pet health assessment
        """
        vision_config = self.vision_prompts.get("full-body-scan", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Weight: {pet_profile.get('weight', 'Unknown')} kg
- Breed: {pet_profile.get('breed', 'Unknown')}
- Medical History: {pet_profile.get('medical_history', 'None reported')}
"""
        
        prompt = f"""You are a {system.get('role', 'veterinary examination specialist')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Thorough, professional, preventative-care focused')}

FULL BODY ASSESSMENT:
{pet_context}
VETERINARY REFERENCE DATABASE:
{rag_context if rag_context else "No additional veterinary information available."}

USER CONCERN/QUESTION:
{user_query}

OUTPUT FORMAT:
## 📊 Overall Condition
[General health appearance and body condition score]

## 🔍 Body System Assessment
- Head/Neck: [Observations]
- Eyes/Ears: [Observations]
- Skin/Coat: [Observations]
- Limbs/Joints: [Observations]
- Gait/Movement: [Observations]
- Abdomen: [Observations]

## ⚠️ Areas of Concern
[Any abnormalities or areas requiring attention]

## 📝 Observations Summary
[Overall assessment of health indicators]

## 🏥 Veterinary Recommendations
[Recommended veterinary evaluation areas]

## 💡 Preventative Care Tips
[Actions to maintain good health]

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 320)} words
- Systematic assessment from head to tail
- Include body condition score
- Note any visible abnormalities"""
        
        return prompt
    
    # ==========================================
    # FOOD ANALYSIS MODEL PROMPT
    # ==========================================
    def build_food_analysis_model_prompt(self, user_query:str, rag_context:str="") -> str:
        """
        Build prompt for Food Safety analysis
        
        Args:
            user_query: User's question about food safety
            rag_context: Retrieved food safety information from RAG
    
        Returns:
        Formatted prompt string for food safety response generation
        """
        route_info =route_food_query(user_query) #get the intent of the query 
        specific_prompt = get_food_response_prompt(route_info['intent']) #get the specific prompt according to the intent
        final_prompt = str(specific_prompt) + "\n\n" + rag_context + "\n\n" + user_query #build the final prompt
        return final_prompt #return the full text
    
    def build_emotion_detection_audio_prompt(self,audio_analysis:str, pet_profile:Dict, rag_context:str) -> str:
        """
        Build prompt for emotion detection from audio analysis
        
        Args:
            audio_analysis: Analysis of audio content
            pet_profile: Pet profile information
            rag_context: Retrieved context from RAG
        
        Returns:
        Formatted prompt string for emotion detection from audio
        """
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Weight: {pet_profile.get('weight', 'Unknown')} kg
- Breed: {pet_profile.get('breed', 'Unknown')}
"""
        
        prompt = f"""You are an expert in analyzing pet emotional states from audio analysis.

Predicted Action of pet from audio provided : {audio_analysis}

retreived context from RAG:
{rag_context if rag_context else "No additional context available."}

Pet's profile information:
{pet_context}

ANALYSIS GUIDELINES:
1. Identify emotional indicators in the audio (e.g., whining, barking, purring)
2. Consider pet's behavior patterns and known traits
3. Assess intensity and duration of emotional expressions
4. Determine if emotions are positive, negative, or neutral

OUTPUT FORMAT:
## 🎵 Emotional Analysis Summary
[General emotional state and mood]

## 🧠 Emotional Indicators Found
- [List specific vocalizations and their emotional significance]

## 📊 Emotional Intensity Score
[Score from 1-10 indicating intensity]

## 📝 Behavioral Context Notes
[Any relevant behavioral context or environmental factors]

CONSTRAINTS:
- Minimum 250 words
- Focus on auditory cues only
- Include specific examples of vocalizations analyzed"""
        
        return prompt
    
    # ==========================================
    # VISION - UNIFIED PROMPT BUILDER
    # ==========================================
    
    def build_vision_prompt(
        self,
        model_type: str,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str = "",
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Unified vision prompt builder that routes to appropriate template
        
        Args:
            model_type: Type of vision model (toy-safety-detection, skin-and-health-diagnostic, etc.)
            predicted_class: Classification result from vision model
            confidence_score: Model confidence (0-1)
            user_query: User's question
            rag_context: Retrieved context from RAG
            pet_profile: Optional pet information
        """
        
        if model_type == "toys-safety-detection":
            return self.build_toy_classifier_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "skin-and-health-diagnostic":
            return self.build_diseases_classifier_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "parasite-detection":
            return self.build_parasite_detection_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "poop-vomit-detection":
            return self.build_poop_vomit_detection_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "home-enviroment-safety-scan":
            return self.build_home_environment_safety_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "packaged-product-scanner":
            return self.build_packaged_product_scanner_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "full-body-scan":
            return self.build_full_body_scan_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "injury-assistance":
            return self.build_injury_assistance_prompt(
                user_query, rag_context
            )
        elif model_type == "pet-food-image-analysis":
            return self.build_food_analysis_model_prompt(
                user_query, rag_context
            )
        elif model_type == "emotion-detection":
            return self.build_emotion_detection_prompt(
                user_query if isinstance(user_query, str) else "",
                "",
                pet_profile or {},
                rag_context
            )
        elif model_type == "poop-vomit-detection":
            return self.build_poop_vomit_detection_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "home-environment-safety-scan":
            return self.build_home_environment_safety_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "emotion-detection-audio":
            return self.build_emotion_detection_audio_prompt(
                audio_analysis = predicted_class, pet_profile= pet_profile or {}, rag_context=rag_context
            )
        else:
            return self.build_vision_default_prompt(
                predicted_class, confidence_score, user_query, rag_context
            )


    # ==========================================
    # RAG-AWARE PROMPT BUILDER
    # ==========================================
    
    def build_rag_aware_prompt(
        self,
        module: str,
        user_query: Dict,
        pet_profile: Dict,
        rag_retrieved_data: str
    ) -> str:
        """
        Build prompt that intelligently uses RAG context
        
        The key to PawPilot: RAG data directly informs the prompt
        Supports all vision detection strategies and emergency scenarios
        """
        if module == "emotion-detection":
            return self.build_emotion_detection_prompt(
                user_query.get("image_features", ""),
                user_query.get("audio_analysis", ""),
                pet_profile,
                rag_retrieved_data
            )
        elif module == "emergency":
            return self.build_emergency_prompt(
                user_query.get("emergency_type", ""),
                user_query.get("symptoms", ""),
                pet_profile,
                rag_retrieved_data
            )
        elif module == "product-safety":
            return self.build_product_analysis_prompt(
                user_query,
                pet_profile,
                rag_retrieved_data
            )
        elif module == "toy-safety-detection":
            return self.build_toy_classifier_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "skin-and-health-diagnostic":
            return self.build_diseases_classifier_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "parasite-detection":
            return self.build_parasite_detection_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "poop-vomit-detection":
            return self.build_poop_vomit_detection_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "home-environment-safety-scan":
            return self.build_home_environment_safety_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "packaged-product-scanner":
            return self.build_packaged_product_scanner_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "full-body-scan":
            return self.build_full_body_scan_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "vision":
            # Generic vision module - auto-detect based on model_type
            return self.build_vision_prompt(
                user_query.get("model_type", "default"),
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "injury-assistance":
            return self.build_injury_assistance_prompt(
                user_query.get("query", ""),
                rag_retrieved_data
            )
        elif module == "pet-food-image-analysis":
            return self.build_food_analysis_model_prompt(
                user_query.get("query", ""),
                rag_retrieved_data
            )
        else:
            raise ValueError(f"Unknown module: {module}")