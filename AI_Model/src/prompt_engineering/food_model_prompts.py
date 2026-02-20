"""
Food Analyzer Prompt Templates
4-Layer Architecture:
  Layer 1: Dynamic Variables - Base templates with runtime placeholders
  Layer 2: Vision Analysis - Specialized prompts for image analysis (4 options)
  Layer 3: Generation Prompts - Response generation after RAG retrieval
  Layer 4: Text Context Detection - Route text-only queries
"""
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LAYER 1: DYNAMIC VARIABLES - Base Templates with Placeholders
# =============================================================================

def get_json_fields_for_food(detail_level: str = "standard") -> str:
    """Get JSON output fields based on detail level"""
    base_fields = {
        "food_items": "list of identified foods",
        "safety_status": "safe/toxic/caution",
        "species": "dog/cat/both",
        "confidence": "0.0 to 1.0"
    }
    
    standard_fields = {
        **base_fields,
        "toxic_items": "list of dangerous foods",
        "toxic_reason": "why each is dangerous",
        "symptoms": "expected symptoms if consumed",
        "urgency": "none/monitor/vet_soon/emergency"
    }
    
    detailed_fields = {
        **standard_fields,
        "toxic_compound": "active toxin (e.g., theobromine)",
        "toxic_dose": "dangerous amount per kg body weight",
        "onset_time": "when symptoms appear",
        "safe_alternatives": "what to feed instead",
        "immediate_action": "what to do right now"
    }
    
    if detail_level == "minimal":
        return str(base_fields)
    elif detail_level == "detailed":
        return str(detailed_fields)
    else:
        return str(standard_fields)


def create_food_analysis_prompt(detail_level: str = "standard") -> ChatPromptTemplate:
    """
    Layer 1: Dynamic prompt template for food safety analysis
    
    Args:
        detail_level: minimal/standard/detailed
        
    Returns:
        ChatPromptTemplate with dynamic variables
    """
    system_template = """You are a pet food safety expert specializing in identifying toxic and safe foods for dogs and cats.
Your role: Analyze food images or descriptions and assess safety for the specified pet species.
Detail Level: {detail_level}
Output Format: JSON only, no markdown, no explanations

Be accurate and conservative with safety assessments - when in doubt, mark as caution."""

    human_template = """Analyze this food for pet safety. Extract the following information:
1. **Food Items**: What foods are visible or described?
2. **Safety Status**: Is this safe, toxic, or requires caution?
3. **Species**: Which pet species is this assessment for?
4. **Toxic Items**: If any, list dangerous foods and why
5. **Urgency**: How urgently should the owner act?

Return ONLY a JSON object with these keys:
{fields}

Pet Species: {species}
Additional Context: {context}"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    chat_prompt = chat_prompt.partial(
        detail_level=detail_level,
        fields=get_json_fields_for_food(detail_level)
    )
    
    return chat_prompt


# =============================================================================
# LAYER 2: VISION ANALYSIS PROMPTS - Specialized for Different Contexts
# =============================================================================

class FoodVisionPrompts:
    """
    Layer 2: Specialized vision prompts for different food analysis scenarios
    
    4 Options:
    A. Basic - Quick food identification
    B. Dog-Specific - Focus on dog-toxic foods
    C. Cat-Specific - Focus on cat-toxic foods  
    D. Multi-Ingredient - Analyze mixed/cooked foods
    """
    
    @staticmethod
    def get_basic_prompt() -> ChatPromptTemplate:
        """
        Option A: Basic food identification
        Use for: Quick safety check, simple foods
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are a professional food analyst with expertise in identifying ingredients.
Your task is to ACCURATELY identify all visible food items in the image.

CRITICAL IDENTIFICATION RULES:
- MEAT IDENTIFICATION: Look at COLOR and TEXTURE carefully:
  * WHITE meat = chicken, turkey, fish
  * BROWN/DARK meat = beef, lamb, pork
  * PINK meat = ham, salmon
- VEGETABLE IDENTIFICATION: Be PRECISE:
  * Broccoli = green florets with tree-like shape
  * Peas = small round green spheres
  * Green beans = long thin green pods
  * Bell pepper = curved strips/chunks
- Do NOT guess - if unsure, say "possibly X or Y"
- Look at the ACTUAL colors, shapes, and textures

Be thorough and ACCURATE - pet safety depends on correct identification."""),
            ("human", """Carefully identify ALL foods visible in this image.

LOOK CLOSELY at:
1. Meat type - check the COLOR (white=chicken, brown=beef, pink=pork)
2. Vegetables - identify by SHAPE (florets=broccoli, rounds=peas, strips=peppers)
3. Grains - rice, pasta, bread, etc.
4. Sauces/liquids - gravy, broth, oil

For each food item, provide:
1. Exact food name (be specific - "beef" not just "meat")
2. Category (protein/vegetable/grain/sauce/other)
3. How you identified it (color, shape, texture)
4. General safety for pets (safe/toxic/unknown)

Return JSON format:
{
  "identified_foods": [
    {"name": "...", "category": "...", "identification_reason": "...", "general_safety": "..."}
  ],
  "dish_description": "Brief description of the overall dish",
  "requires_detailed_analysis": true/false,
  "notes": "any concerns or observations"
}""")
        ])
    
    @staticmethod
    def get_dog_specific_prompt() -> ChatPromptTemplate:
        """
        Option B: Dog-specific food analysis
        Focus on: Chocolate, grapes, xylitol, onions, macadamia nuts
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing food for a DOG. Critical toxins to detect:
- Chocolate/cocoa (theobromine toxicity)
- Grapes/raisins (kidney failure)
- Xylitol/artificial sweeteners (hypoglycemia, liver failure)
- Onions/garlic/leeks/chives (hemolytic anemia)
- Macadamia nuts (weakness, tremors)
- Alcohol (ethanol toxicity)
- Avocado (persin toxicity)
- Coffee/caffeine (cardiac issues)

ANY amount of these is dangerous. Be extremely vigilant."""),
            ("human", """Analyze this food image specifically for DOG safety.

Critical checks:
1. Any chocolate or cocoa products?
2. Any grapes, raisins, or grape products?
3. Any sugar-free items (may contain xylitol)?
4. Any onions, garlic, or allium family?
5. Any nuts, especially macadamia?
6. Any alcohol or coffee?

Return JSON:
{
  "species": "dog",
  "identified_foods": [...],
  "toxic_items_found": [...],
  "toxic_compounds_detected": [...],
  "safety_status": "safe/toxic/caution",
  "urgency": "none/monitor/vet_soon/emergency",
  "immediate_action": "what owner should do"
}""")
        ])
    
    @staticmethod
    def get_cat_specific_prompt() -> ChatPromptTemplate:
        """
        Option C: Cat-specific food analysis
        Focus on: Lilies, essential oils, onions, raw fish, dairy
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing food for a CAT. Critical toxins to detect:
- Lilies (ALL parts - extreme kidney toxicity, often fatal)
- Essential oils (liver damage, neurological issues)
- Onions/garlic/leeks/chives (hemolytic anemia - cats MORE sensitive than dogs)
- Raw fish (thiamine deficiency)
- Dairy products (lactose intolerance common)
- Grapes/raisins (kidney issues - though less studied in cats)
- Chocolate (theobromine - cats rarely eat but still toxic)
- Caffeine (cardiac issues)

Cats are obligate carnivores - many human foods are unsuitable."""),
            ("human", """Analyze this food image specifically for CAT safety.

Critical checks:
1. Any lily flowers or plants visible?
2. Any essential oils or scented products?
3. Any onions, garlic, or related foods?
4. Any raw fish or seafood?
5. Any dairy products?
6. Any chocolate, coffee, or caffeinated items?

Return JSON:
{
  "species": "cat",
  "identified_foods": [...],
  "toxic_items_found": [...],
  "toxic_compounds_detected": [...],
  "safety_status": "safe/toxic/caution",
  "urgency": "none/monitor/vet_soon/emergency",
  "immediate_action": "what owner should do"
}""")
        ])
    
    @staticmethod
    def get_mixed_meal_prompt() -> ChatPromptTemplate:
        """
        Option D: Multi-ingredient/cooked food analysis
        For: Prepared meals, leftovers, mixed dishes
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are a professional food analyst and pet nutrition expert.
You must ACCURATELY identify all ingredients in cooked/prepared foods.

CRITICAL IDENTIFICATION RULES:

MEAT IDENTIFICATION (by COLOR):
- BROWN/DARK chunks = BEEF, lamb, or dark meat
- WHITE meat = chicken, turkey, fish  
- PINK meat = pork, ham, salmon
- DO NOT confuse beef (brown) with chicken (white)

VEGETABLE IDENTIFICATION (by SHAPE):
- BROCCOLI = green tree-like florets with stems
- PEAS = tiny round green balls (much smaller than other veggies)
- GREEN BEANS = long thin green pods
- BELL PEPPER = curved strips or chunks
- CARROTS = orange rounds or sticks
- SWEET POTATO = orange cubes (darker orange than carrots)
- PUMPKIN = pale orange, often mashed

SAUCE IDENTIFICATION:
- Brown sauce/gravy = soy-based, beef-based, or mushroom
- Clear liquid = broth
- White sauce = cream-based

Common hidden toxins in human food:
- Onions/garlic in sauces, gravies, seasonings
- Xylitol in sugar-free items
- High salt/fat (pancreatitis risk)
- Spices (nutmeg, excess salt)

When unsure, say "possibly X" rather than guessing wrong."""),
            ("human", """Analyze this prepared/mixed food for pet safety.

STEP 1 - PRECISE IDENTIFICATION:
Look at EACH ingredient carefully:
- What COLOR is the meat? (brown=beef, white=chicken)
- What SHAPE are the vegetables? (florets=broccoli, rounds=peas)
- Is there sauce? What color/type?

STEP 2 - LIST ALL VISIBLE INGREDIENTS:
Be specific! Say "beef chunks" not "meat", say "broccoli florets" not "green vegetable"

STEP 3 - CONSIDER HIDDEN INGREDIENTS:
What seasonings/sauces might be in this dish?

Return JSON:
{
  "dish_type": "specific description (e.g., 'Beef stir-fry with vegetables')",
  "visible_ingredients": [
    {"name": "...", "identification_reason": "brown chunks = beef", "category": "protein/vegetable/grain/sauce"}
  ],
  "likely_hidden_ingredients": ["garlic", "soy sauce", "salt", etc.],
  "toxic_concerns": ["list any ingredients toxic to pets"],
  "safety_status": "safe/toxic/caution",
  "species_safety": {
    "dog": "safe/toxic/caution",
    "cat": "safe/toxic/caution"
  },
  "recommendation": "specific guidance for pet owner",
  "safe_portion": "if any part is safe, how much"
}""")
        ])


# =============================================================================
# LAYER 3: GENERATION PROMPTS - Response After RAG Retrieval
# =============================================================================

class FoodSafetyResponsePrompts:
    """
    Layer 3: Generate final responses using retrieved knowledge
    """
    
    @staticmethod
    def get_safety_response_prompt() -> ChatPromptTemplate:
        """Generate comprehensive safety response"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a pet nutrition expert providing food safety guidance.
Use the retrieved knowledge base information to give accurate, helpful advice.
Be clear about urgency levels and always err on side of caution.
If emergency symptoms described, always recommend immediate vet care."""),
            ("human", """Based on the analysis and knowledge base:

Identified Food: {food_items}
Pet Species: {species}
Retrieved Knowledge: {retrieved_context}

Provide a complete safety assessment:
1. Safety Status (safe/toxic/caution)
2. If toxic: specific dangers and symptoms
3. What to do RIGHT NOW
4. When to see a vet
5. Safe alternatives they could feed instead

Be helpful but prioritize safety. Format as clear, actionable guidance.""")
        ])
    
    @staticmethod
    def get_emergency_response_prompt() -> ChatPromptTemplate:
        """Emergency situation - pet already consumed something"""
        return ChatPromptTemplate.from_messages([
            ("system", """EMERGENCY PROTOCOL ACTIVATED.
The pet has already consumed potentially toxic food.
Provide immediate, clear, actionable guidance.
Time is critical - be concise and direct.
Always recommend calling vet/poison control for serious toxins."""),
            ("human", """URGENT: Pet has consumed potentially toxic food.

What was consumed: {food_consumed}
Pet type: {species}
Amount consumed: {amount}
Pet weight: {weight}
Time since consumption: {time_elapsed}
Current symptoms: {symptoms}

Retrieved Knowledge: {retrieved_context}

Provide IMMEDIATE guidance:
1. Is this an emergency? (yes/no/maybe)
2. What to do RIGHT NOW (specific steps)
3. Should they call vet/poison control? (phone numbers if applicable)
4. What symptoms to watch for
5. What NOT to do (don't induce vomiting for X, etc.)

Be direct and actionable. This is urgent.""")
        ])
    
    @staticmethod
    def get_recommendation_prompt() -> ChatPromptTemplate:
        """Safe food recommendations"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are recommending safe, healthy foods for pets.
Use knowledge base to suggest nutritious options.
Consider species, provide portion guidance, mention preparation tips."""),
            ("human", """User wants food recommendations for their pet.

Pet Species: {species}
Pet Size: {size}
Request: {request}
Any Restrictions: {restrictions}

Retrieved Knowledge: {retrieved_context}

Provide helpful recommendations:
1. Top 3-5 safe food options
2. Nutritional benefits of each
3. Proper portion sizes for their pet size
4. How to prepare/serve
5. Any cautions or moderation advice

Be helpful and informative. Focus on healthy, safe options.""")
        ])


# =============================================================================
# LAYER 4: TEXT CONTEXT DETECTION - Route Text-Only Queries
# =============================================================================

class FoodQueryRouter:
    """
    Layer 4: Detect intent from text queries and route appropriately
    """
    
    QUERY_PATTERNS = {
        "safety_check": [
            "can dogs eat", "can cats eat", "is it safe", "safe for dogs",
            "safe for cats", "toxic to dogs", "toxic to cats", "okay to give",
            "feed my dog", "feed my cat", "give my pet"
        ],
        "emergency": [
            "ate", "consumed", "swallowed", "licked", "chewed", "got into",
            "accidentally", "just ate", "help my dog ate", "my cat ate",
            "poisoned", "emergency", "throwing up", "vomiting"
        ],
        "recommendation": [
            "best food for", "healthy treats", "what can i feed",
            "recommend", "suggestion", "good for dogs", "good for cats",
            "nutritious", "healthy snacks", "treats for"
        ],
        "information": [
            "why is", "what makes", "how much", "toxic because",
            "dangerous", "symptoms of", "signs of poisoning"
        ]
    }
    
    @staticmethod
    def detect_intent(query: str) -> Dict:
        """
        Detect query intent from text
        
        Returns:
            Dict with intent type, confidence, detected patterns
        """
        query_lower = query.lower()
        
        matches = {}
        for intent, patterns in FoodQueryRouter.QUERY_PATTERNS.items():
            count = sum(1 for p in patterns if p in query_lower)
            if count > 0:
                matches[intent] = count
        
        if not matches:
            return {
                "intent": "general",
                "confidence": 0.5,
                "patterns_matched": [],
                "prompt_to_use": "basic"
            }
        
        # Handle priority: questions with "?" are usually safety_check, not emergency
        is_question = "?" in query or query_lower.startswith(("is", "can", "are", "does", "should"))
        
        # If it's a question and matches both safety_check and emergency,
        # prefer safety_check unless has strong emergency words
        strong_emergency_words = ["just ate", "help", "poisoned", "throwing up", "vomiting", "emergency"]
        has_strong_emergency = any(w in query_lower for w in strong_emergency_words)
        
        if is_question and "safety_check" in matches and "emergency" in matches:
            if not has_strong_emergency:
                # Boost safety_check for questions
                matches["safety_check"] += 2
        
        primary_intent = max(matches, key=matches.get)
        
        # Map intent to prompt
        prompt_map = {
            "safety_check": "species_specific",
            "emergency": "emergency",
            "recommendation": "recommendation",
            "information": "basic"
        }
        
        return {
            "intent": primary_intent,
            "confidence": min(0.9, 0.5 + (matches[primary_intent] * 0.2)),
            "patterns_matched": [p for p in FoodQueryRouter.QUERY_PATTERNS[primary_intent] 
                                 if p in query_lower],
            "prompt_to_use": prompt_map.get(primary_intent, "basic")
        }
    
    @staticmethod
    def detect_species(query: str) -> str:
        """Detect pet species from query"""
        query_lower = query.lower()
        
        dog_words = ["dog", "puppy", "pup", "canine"]
        cat_words = ["cat", "kitten", "kitty", "feline"]
        
        has_dog = any(w in query_lower for w in dog_words)
        has_cat = any(w in query_lower for w in cat_words)
        
        if has_dog and not has_cat:
            return "dog"
        elif has_cat and not has_dog:
            return "cat"
        elif has_dog and has_cat:
            return "both"
        else:
            return "unknown"


# =============================================================================
# UTILITY FUNCTIONS - Prompt Selection and Integration
# =============================================================================

def get_food_vision_prompt(context: str = "basic", species: str = "unknown") -> ChatPromptTemplate:
    """
    Get appropriate vision prompt based on context and species
    
    Args:
        context: basic/dog/cat/mixed
        species: dog/cat/both/unknown
        
    Returns:
        Appropriate ChatPromptTemplate
    """
    if context == "mixed":
        return FoodVisionPrompts.get_mixed_meal_prompt()
    elif species == "dog" or context == "dog":
        return FoodVisionPrompts.get_dog_specific_prompt()
    elif species == "cat" or context == "cat":
        return FoodVisionPrompts.get_cat_specific_prompt()
    else:
        return FoodVisionPrompts.get_basic_prompt()


def get_response_prompt(intent: str = "safety") -> ChatPromptTemplate:
    """
    Get appropriate response generation prompt
    
    Args:
        intent: safety/emergency/recommendation
        
    Returns:
        Appropriate ChatPromptTemplate
    """
    if intent == "emergency":
        return FoodSafetyResponsePrompts.get_emergency_response_prompt()
    elif intent == "recommendation":
        return FoodSafetyResponsePrompts.get_recommendation_prompt()
    else:
        return FoodSafetyResponsePrompts.get_safety_response_prompt()


def route_food_query(query: str, has_image: bool = False) -> Dict:
    """
    Main routing function for food queries
    
    Args:
        query: User's text query
        has_image: Whether an image was provided
        
    Returns:
        Dict with routing information
    """
    intent_info = FoodQueryRouter.detect_intent(query)
    species = FoodQueryRouter.detect_species(query)
    
    # Determine vision prompt if image provided
    if has_image:
        if "mixed" in query.lower() or "cooked" in query.lower() or "meal" in query.lower():
            vision_context = "mixed"
        else:
            vision_context = species if species in ["dog", "cat"] else "basic"
    else:
        vision_context = None
    
    return {
        **intent_info,
        "species": species,
        "has_image": has_image,
        "vision_context": vision_context,
        "vision_prompt": get_food_vision_prompt(vision_context, species) if has_image else None,
        "response_prompt": get_response_prompt(intent_info["intent"])
    }


# For backward compatibility and easy access
PROMPTS = {
    "food_analysis_base": create_food_analysis_prompt,
    "vision_basic": FoodVisionPrompts.get_basic_prompt,
    "vision_dog": FoodVisionPrompts.get_dog_specific_prompt,
    "vision_cat": FoodVisionPrompts.get_cat_specific_prompt,
    "vision_mixed": FoodVisionPrompts.get_mixed_meal_prompt,
    "response_safety": FoodSafetyResponsePrompts.get_safety_response_prompt,
    "response_emergency": FoodSafetyResponsePrompts.get_emergency_response_prompt,
    "response_recommendation": FoodSafetyResponsePrompts.get_recommendation_prompt,
}
