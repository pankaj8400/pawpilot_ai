"""
# Vision Prompt Templates - LangChain-based dynamic prompts for Product Analysis
# Supports 4 approaches: Basic, Specialized, Few-Shot, Conditional
# """
from langchain_core.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate) 
from langchain.messages import SystemMessage, HumanMessage #to make the model recognize the roles
from typing import Dict, List, Optional


def get_json_fields_for_level(detail_level:str) -> str:
    """
    Generate JSON field structure based on detail level
    
    Args:
        detail_level: "basic" | "standard" | "detailed"
        
    Returns:
        JSON string with appropriate fields
        
    Example:
        >>> get_json_fields_for_level("basic")
        '    "product_name": "...",\\n    "product_type": "..."'
    """
    # Base fields (always include)
    base_fields = [
        '"product_name": "..."',
        '"product_type": "..."'
    ]
    
    # Standard adds species and category
    standard_fields = base_fields + [
        '"target_species": "..."',
        '"category": "..."',
        '"brand": "..."'
    ]
    
    # Detailed adds ingredient and quality observations
    detailed_fields = standard_fields + [
        '"visible_ingredients": ["..."]',
        '"nutritional_claims": ["..."]',
        '"age_recommendation": "..."',
        '"quality_indicators": "..."'
    ]
    
    # Select fields based on level
    if detail_level == "basic":
        fields = base_fields
    elif detail_level == "detailed":
        fields = detailed_fields
    else:  # standard (default)
        fields = standard_fields
    
    # Format as JSON structure
    return "{\n    " + ",\n    ".join(fields) + "\n}"


# =====================================================
# OPTION 1: Basic Dynamic Template (Production Ready)
# =====================================================

def create_product_analysis_prompt(detail_level:str = "standard") -> ChatPromptTemplate:
    """
    Create a dynamic vision analysis prompt for pet products

    Args:
        detail_level: "basic" | "standard" | "detailed"
    
    Returns: ChatPromptTemplate ready for vision model
    Example:
        >>> prompt = create_product_analysis_prompt("detailed")
        >>> messages = prompt.format_messages(fields="...", additional_instructions="...")
    """

    # Define system template (AI's role)
    system_template = """You are a pet product analysis assistant specializing in ingredient assessment and nutritional evaluation.
    Your role: Analyze pet product images and extract structured information for product recommendations.
    Detail Level : {detail_level}
    Output Format: JSON only, no markdown, no explanations

    Be accurate and focus on pet health and safety."""

    human_template = """Analyze this pet product image carefully. Extract the following information:
1. **Product Name**: What is the product name/brand visible?
2. **Product Type**: What type of product is this? (treat, dry_food, wet_food, supplement, shampoo, toy, litter, etc.)
3. **Target Species**: Who is this product for? (dog, cat, both)
4. **Category**: Product category (treats, dry_food, supplements, grooming, litter, toys)
5. **Brand**: What brand is visible on the packaging?
Return ONLY a JSON object with these keys:
{fields}

Additional Instructions: {additional_instructions}"""

    # Create the ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    # Create default values for template variables
    # This makes the template usable with .format_messages()
    chat_prompt = chat_prompt.partial(
        detail_level=detail_level,
        fields=get_json_fields_for_level(detail_level),
        additional_instructions="Provide accurate assessment based on visible packaging and labels."
    )

    return chat_prompt

# =====================================================
# Backward Compatibility - Default Prompt
# =====================================================

# Create a default prompt for existing code that imports "prompt"
# This maintains compatibility with vision_integration.py

_default_template = create_product_analysis_prompt(detail_level="standard")
_default_messages = _default_template.format_messages() # will fill the dynamic variable details

# Extract the human message text (what the vision model actually sees)
prompt = _default_messages[1].content

# For advanced users who want the full template:
prompt_template = create_product_analysis_prompt 

# =====================================================
# OPTION 2: Specialized Prompts (Different Scenarios)
# =====================================================

class VisionPromptTemplates:
    """
    Collection of specialized prompt templates for different product scenarios
    Each method returns a ChatPromptTemplate optimized for that specific case
    """

    @staticmethod
    def get_treat_analysis_prompt() -> ChatPromptTemplate:
        """
        Specialized template for PET TREATS analysis
        Focus: Ingredients, nutritional value, allergens, safety
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a PET TREAT product. Critical focus areas:
- Ingredient quality (whole ingredients vs fillers vs by-products)
- Protein source identification (chicken, beef, fish, etc.)
- Harmful ingredients detection (xylitol, artificial colors, BHA/BHT)
- Allergen identification (common: poultry, beef, grains, dairy)
- Nutritional claims on packaging

Be thorough with ingredient analysis - pet treat quality varies greatly."""),
            ("human", """Analyze this PET TREAT product image.

Extract and assess:
1. Product name and brand
2. Target species (dog/cat) and any size/age recommendations
3. Primary protein source visible
4. Any visible ingredient warnings or allergen info
5. Nutritional claims (grain-free, limited ingredient, etc.)
6. Quality indicators (AAFCO statement, country of origin)

Return JSON format with treat-specific analysis fields.""")
        ])

    @staticmethod
    def get_food_analysis_prompt() -> ChatPromptTemplate:
        """
        Specialized template for PET FOOD analysis (dry/wet)
        Focus: Nutritional completeness, protein content, AAFCO compliance
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a PET FOOD product (dry or wet food). Critical focus areas:
- AAFCO nutritional adequacy statement
- First 5 ingredients (most important!)
- Protein percentage and source
- Life stage appropriateness (puppy/kitten, adult, senior, all life stages)
- Grain-free concerns (FDA DCM investigation for dogs)
- By-products vs whole meat assessment

Pet food is the foundation of health - be thorough and accurate."""),
            ("human", """Analyze this PET FOOD product image.

Extract and assess:
1. Product name, brand, and formula type
2. Target species and life stage (puppy, adult, senior, all)
3. First visible ingredients (up to 5)
4. Guaranteed analysis if visible (protein %, fat %, fiber %)
5. Special diet claims (grain-free, limited ingredient, weight management)
6. AAFCO statement if visible

Return JSON format with food-specific nutritional fields.""")
        ])

    @staticmethod
    def get_supplement_analysis_prompt() -> ChatPromptTemplate:
        """
        Specialized template for PET SUPPLEMENTS analysis
        Focus: Active ingredients, dosage, quality seals, intended purpose
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a PET SUPPLEMENT product. Critical focus areas:
- Active ingredient identification (glucosamine, omega-3, probiotics, etc.)
- Dosage information visibility
- Quality seals (NASC, third-party testing)
- Intended purpose (joint, skin/coat, digestive, calming, etc.)
- Species and size appropriateness
- Form factor (chew, powder, liquid, capsule)

Supplements vary widely in quality - look for quality indicators."""),
            ("human", """Analyze this PET SUPPLEMENT product image.

Extract and assess:
1. Product name and brand
2. Supplement type/purpose (joint, digestive, skin, calming, etc.)
3. Active ingredients if visible
4. Target species and any size/weight specifications
5. Quality seals or certifications visible (NASC, etc.)
6. Form (soft chew, tablet, powder, liquid)

Return JSON format with supplement-specific fields.""")
        ])

    @staticmethod
    def get_grooming_product_prompt() -> ChatPromptTemplate:
        """
        Specialized template for PET GROOMING products (shampoos, sprays, creams)
        Focus: Ingredients safety, skin sensitivity, intended use
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a PET GROOMING product. Critical focus areas:
- Ingredient safety for pets (no tea tree oil in high concentrations!)
- pH balance for pet skin (different from human)
- Intended use (regular bathing, medicated, flea/tick, sensitive skin)
- Harsh chemical detection (sulfates, parabens, artificial fragrances)
- Natural vs synthetic ingredient balance
- Species-specific formulation (dog vs cat - cats groom themselves!)

Cats are especially sensitive to grooming product ingredients."""),
            ("human", """Analyze this PET GROOMING product image.

Extract and assess:
1. Product name and brand
2. Product type (shampoo, conditioner, spray, balm, ear cleaner)
3. Target species (dog/cat/both)
4. Intended purpose (regular, medicated, sensitive, flea treatment)
5. Key beneficial ingredients visible (oatmeal, aloe, etc.)
6. Any warning labels or usage instructions visible

Return JSON format with grooming-specific fields.""")
        ])


# =====================================================
# OPTION 3: Few-Shot Learning (Improved Accuracy)
# =====================================================

def create_fewshot_vision_prompt() -> ChatPromptTemplate:
    """
    Use few-shot examples to teach the AI proper output format

    Benefits:
    - More consistent JSON structure
    - Better handling of edge cases 
    - Higher accuracy on field values

    Return:
        ChatPromptTemplate with embedded examples
    """

    # Define example inputs and ideal output
    examples = [
        {
            "description": "Dog treat bag with chicken jerky, shows 'Made in USA' and 'Grain Free'",
            "output": '{"product_name": "Chicken Jerky Strips", "product_type": "treat", "target_species": "dog", "category": "treats", "brand": "visible_brand", "quality_indicators": "Made in USA, Grain Free"}'
        },
        {
            "description": "Cat food bag showing salmon flavor, 40% protein, for adult cats",
            "output": '{"product_name": "Salmon Recipe Adult Cat Food", "product_type": "dry_food", "target_species": "cat", "category": "dry_food", "brand": "visible_brand", "life_stage": "adult"}'
        },
        {
            "description": "Joint supplement bottle for dogs with glucosamine and chondroitin, NASC seal visible",
            "output": '{"product_name": "Joint Health Plus", "product_type": "supplement", "target_species": "dog", "category": "supplements", "brand": "visible_brand", "quality_indicators": "NASC certified"}'
        },
        {
            "description": "Oatmeal dog shampoo bottle, says 'For Sensitive Skin' and 'Soap-Free'",
            "output": '{"product_name": "Oatmeal Soothing Shampoo", "product_type": "shampoo", "target_species": "dog", "category": "grooming", "brand": "visible_brand", "intended_use": "sensitive skin"}'
        }
    ]
    
    # Create example prompt template
    # This defines how each example is shown
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Image shows: {description}"),
        ("ai", "{output}")
    ])
    
    # This combines all examples together 
    from langchain_core.prompts import FewShotChatMessagePromptTemplate

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    # Create final prompt with examples + new request
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a pet product analysis assistant.
Learn from these example analyses, then analyze the new product image following the EXACT same format."""),
        few_shot_prompt,  # Inserts all examples here
        ("human", "Now analyze this NEW product image and provide JSON in the same format.")
    ])

    return final_prompt

# =====================================================
# OPTION 4: Conditional Prompt Selection (Smart Router)
# =====================================================

def get_prompt_for_context(product_context: Optional[Dict] = None) -> ChatPromptTemplate:
    """
    Intelligently select the best prompt template based on context
    
    This is the "smart router" - it chooses the optimal template based on
    what you know about the product before analysis.
    
    Args:
        product_context: Dictionary with optional keys:
            - "product_type": "treat" | "food" | "supplement" | "grooming" | None
            - "target_species": "dog" | "cat" | None  
            - "detail_level": "basic" | "standard" | "detailed"
            - "use_examples": True | False (use few-shot)
            
    Returns:
        The most appropriate ChatPromptTemplate
        
    Examples:
        >>> # Dog treat - use specialized treat template
        >>> prompt = get_prompt_for_context({"product_type": "treat"})
        
        >>> # Supplement - use supplement template
        >>> prompt = get_prompt_for_context({"product_type": "supplement"})
        
        >>> # Need high accuracy - use few-shot
        >>> prompt = get_prompt_for_context({"use_examples": True})
        
        >>> # Unknown product - use basic template
        >>> prompt = get_prompt_for_context()
    """
    # Default to empty context if none provided
    if product_context is None:
        product_context = {}
    
    # Create templates instance for Option 2
    templates = VisionPromptTemplates()
    
    # Priority 1: Use few-shot if requested (highest accuracy)
    if product_context.get("use_examples"):
        return create_fewshot_vision_prompt()
    
    # Priority 2: Check for specialized scenarios (Option 2)
    product_type = product_context.get("product_type", "").lower()
    
    # Route to specialized templates based on product type
    if "treat" in product_type or "snack" in product_type or "chew" in product_type:
        return templates.get_treat_analysis_prompt()
    
    if "food" in product_type or "kibble" in product_type:
        return templates.get_food_analysis_prompt()
    
    if "supplement" in product_type or "vitamin" in product_type or "joint" in product_type:
        return templates.get_supplement_analysis_prompt()
    
    if "shampoo" in product_type or "grooming" in product_type or "spray" in product_type:
        return templates.get_grooming_product_prompt()
    
    # Priority 3: Use Option 1 with specified detail level
    detail_level = product_context.get("detail_level", "standard")
    return create_product_analysis_prompt(detail_level)


# =====================================================
# OPTION 5: Product Analysis Generation Prompts (NEW)
# =====================================================

class ProductAnalysisGenerationPrompts:
    """
    Specialized prompts for generating product analysis and recommendations
    Mirrors the vision prompt specialization for consistent expertise
    
    Usage: Use same prompt strategy for both vision analysis and recommendation generation
    """
    
    @staticmethod
    def get_treat_analysis_generation_prompt() -> str:
        """
        Specialized prompt for generating PET TREAT analysis and recommendations
        
        Focus areas:
        - Ingredient quality assessment
        - Nutritional value scoring
        - Allergen identification
        - Safety concerns
        - Better alternatives
        
        Returns:
            System instruction string for LLM
        """
        return """You are a pet nutrition expert specializing in PET TREATS analysis.

CRITICAL FOCUS AREAS:
- Ingredient quality (whole ingredients vs fillers vs by-products)
- Protein source quality (real meat vs meat meal vs by-products)
- Harmful ingredient detection (xylitol, artificial colors, BHA/BHT, propylene glycol)
- Allergen identification (common: poultry, beef, grains, dairy)
- Caloric density and feeding recommendations
- Age and breed suitability

ANALYSIS STYLE:
- Rate nutritional value 1-10 with clear reasoning
- List top 3 PROS of this product
- List top 3 CONS or concerns
- Identify any WARNING ingredients
- Suggest better alternatives if quality is low
- Consider price/value ratio
- Be specific about which pets this is best/worst for

Based on the product protocols provided, generate a comprehensive treat analysis."""

    @staticmethod
    def get_food_analysis_generation_prompt() -> str:
        """
        Specialized prompt for generating PET FOOD analysis and recommendations
        
        Focus areas:
        - AAFCO nutritional adequacy
        - Ingredient quality (first 5 ingredients)
        - Protein content and source
        - Life stage appropriateness
        - Special dietary considerations
        
        Returns:
            System instruction string for LLM
        """
        return """You are a pet nutrition expert specializing in PET FOOD analysis.

CRITICAL FOCUS AREAS:
- AAFCO nutritional adequacy (complete and balanced?)
- First 5 ingredients analysis (most important!)
- Protein percentage and quality of protein sources
- Life stage appropriateness (puppy/kitten, adult, senior, all life stages)
- Grain-free concerns (FDA DCM investigation for dogs - mention if relevant)
- By-products assessment (not always bad, but quality varies)
- Carbohydrate content and sources

ANALYSIS STYLE:
- Rate overall nutritional quality 1-10 with reasoning
- Analyze the first 5 ingredients in detail
- Identify the primary protein source and its quality
- Note any concerning ingredients
- Assess value for price point
- Recommend which pets this food is BEST suited for
- Recommend which pets should AVOID this food
- Suggest comparable or better alternatives if needed

Based on the product protocols provided, generate a comprehensive food analysis."""

    @staticmethod
    def get_supplement_analysis_generation_prompt() -> str:
        """
        Specialized prompt for generating PET SUPPLEMENT analysis and recommendations
        
        Focus areas:
        - Active ingredient effectiveness
        - Dosage appropriateness
        - Quality certifications
        - Evidence-based assessment
        - Value for money
        
        Returns:
            System instruction string for LLM
        """
        return """You are a veterinary supplement expert specializing in PET SUPPLEMENTS analysis.

CRITICAL FOCUS AREAS:
- Active ingredient identification and therapeutic doses
- Scientific evidence for claimed benefits
- Quality seals (NASC certification is gold standard)
- Appropriate dosing for pet size
- Potential interactions or contraindications
- Form factor suitability (some pets won't take pills)
- Comparison to veterinary-grade alternatives

ANALYSIS STYLE:
- Rate supplement quality 1-10 based on ingredients and certifications
- Assess if active ingredient doses are therapeutic (enough to work)
- Evaluate scientific support for health claims
- Identify quality certifications present
- Note any missing quality indicators
- Recommend pet types that would benefit most
- Warn about pets that should avoid (e.g., pregnant, on medications)
- Suggest prescription alternatives for severe cases

Based on the product protocols provided, generate a comprehensive supplement analysis."""

    @staticmethod
    def get_grooming_analysis_generation_prompt() -> str:
        """
        Specialized prompt for generating PET GROOMING product analysis
        
        Focus areas:
        - Ingredient safety for pets
        - pH appropriateness
        - Effectiveness for intended use
        - Skin sensitivity considerations
        - Species-specific concerns
        
        Returns:
            System instruction string for LLM
        """
        return """You are a pet dermatology expert specializing in PET GROOMING product analysis.

CRITICAL FOCUS AREAS:
- Ingredient safety (some human-safe ingredients are toxic to pets!)
- pH balance (pet skin pH differs from humans: dogs 6.2-7.4, cats 6.0-7.0)
- Harsh chemical detection (sulfates, parabens, artificial fragrances)
- Species-specific concerns (cats groom themselves and ingest products!)
- Skin condition appropriateness (dry skin, allergies, infections)
- Essential oil safety (many are toxic to cats!)

DANGER INGREDIENTS TO FLAG:
- Tea tree oil (toxic in concentrations over 1-2%)
- Phenols and pine oils (toxic to cats)
- Permethrin (deadly to cats, safe for dogs)
- High concentrations of essential oils

ANALYSIS STYLE:
- Rate product safety 1-10 for intended species
- Identify beneficial ingredients (oatmeal, aloe, etc.)
- Flag any concerning ingredients with explanation
- Assess effectiveness for stated purpose
- Note frequency of use recommendations
- Warn about species it should NOT be used on
- Suggest alternatives for sensitive skin cases

Based on the product protocols provided, generate a comprehensive grooming product analysis."""

    @staticmethod
    def get_toy_analysis_generation_prompt() -> str:
        """
        Specialized prompt for generating PET TOY safety analysis
        
        Focus areas:
        - Material safety
        - Durability assessment
        - Choking hazard evaluation
        - Size appropriateness
        - Supervision requirements
        
        Returns:
            System instruction string for LLM
        """
        return """You are a pet safety expert specializing in PET TOY analysis.

CRITICAL FOCUS AREAS:
- Material safety (BPA-free, non-toxic dyes, food-grade materials)
- Durability for chew strength (aggressive chewers need durable toys)
- Choking hazard assessment (small parts, stuffing, squeakers)
- Size appropriateness (too small = choking risk)
- Supervision requirements
- Digestibility if pieces are swallowed

SAFETY CONCERNS TO FLAG:
- Small removable parts
- Stuffing that can be ingested
- Squeakers that can be swallowed
- Materials that splinter (some plastics, sticks)
- Strings or ribbons (intestinal blockage risk)

ANALYSIS STYLE:
- Rate safety 1-10 for intended pet size/chew strength
- Identify material type and safety
- Assess durability for different chewer types
- Note supervision requirements
- Recommend appropriate pet sizes/types
- Warn about pets that should avoid (aggressive chewers, etc.)
- Suggest safer alternatives if concerns exist

Based on the product protocols provided, generate a comprehensive toy safety analysis."""

    @staticmethod
    def get_litter_analysis_generation_prompt() -> str:
        """
        Specialized prompt for generating CAT LITTER analysis
        
        Focus areas:
        - Dust levels (respiratory health)
        - Clumping effectiveness
        - Odor control
        - Tracking
        - Environmental impact
        
        Returns:
            System instruction string for LLM
        """
        return """You are a cat care expert specializing in CAT LITTER analysis.

CRITICAL FOCUS AREAS:
- Dust levels (critical for respiratory health - cats and humans)
- Clumping ability and ease of scooping
- Odor control effectiveness
- Tracking outside the box
- Safety if ingested (kittens especially)
- Environmental impact and disposal
- Texture preference (some cats are picky!)

LITTER TYPES TO UNDERSTAND:
- Clay clumping: Effective but dusty, not eco-friendly
- Silica crystal: Low dust, long-lasting, some cats dislike texture
- Natural (walnut, corn, wheat, paper): Eco-friendly, varies in performance
- Pine/wood: Good odor control, may track

ANALYSIS STYLE:
- Rate overall quality 1-10 for typical use
- Assess dust levels (critical for asthmatic cats/owners)
- Evaluate clumping and odor control
- Note tracking tendency
- Consider multi-cat household suitability
- Mention environmental/disposal factors
- Recommend cat types this is best/worst for
- Suggest alternatives based on common complaints

Based on the product protocols provided, generate a comprehensive litter analysis."""


# =====================================================
# OPTION 6: Text Query Context Detection (NEW)
# =====================================================

def detect_product_context_from_text(query: str) -> dict:
    """
    Analyze text query to automatically determine product context
    Used for text-only queries when no vision analysis is available
    
    This enables the same specialized prompting for text queries as image queries!
    
    Args:
        query: User's text description of product query
        
    Returns:
        Dictionary with product context:
        {
            "product_type": "treat" | "food" | "supplement" | "grooming" | "toy" | "litter" | None,
            "target_species": "dog" | "cat" | None,
            "detail_level": "standard"
        }
        
    Example:
        >>> detect_product_context_from_text("is this dog treat healthy")
        {"product_type": "treat", "target_species": "dog", "detail_level": "standard"}
        
        >>> detect_product_context_from_text("cat food high protein")
        {"product_type": "food", "target_species": "cat", "detail_level": "standard"}
    """
    query_lower = query.lower()
    
    context = {
        "product_type": None,
        "target_species": None,
        "detail_level": "standard"
    }
    
    # Detect product type (priority order based on specificity)
    if any(word in query_lower for word in ["treat", "snack", "chew", "jerky", "biscuit", "cookie"]):
        context["product_type"] = "treat"
    elif any(word in query_lower for word in ["food", "kibble", "wet food", "dry food", "meal", "diet"]):
        context["product_type"] = "food"
    elif any(word in query_lower for word in ["supplement", "vitamin", "joint", "glucosamine", "probiotic", "omega", "fish oil"]):
        context["product_type"] = "supplement"
    elif any(word in query_lower for word in ["shampoo", "conditioner", "grooming", "spray", "balm", "cream", "ear clean"]):
        context["product_type"] = "grooming"
    elif any(word in query_lower for word in ["toy", "ball", "chew toy", "puzzle", "interactive", "plush", "squeaky"]):
        context["product_type"] = "toy"
    elif any(word in query_lower for word in ["litter", "cat litter", "clumping", "crystal", "silica"]):
        context["product_type"] = "litter"
    
    # Detect target species
    dog_keywords = ["dog", "puppy", "canine", "pup", "doggy", "dogs"]
    cat_keywords = ["cat", "kitten", "feline", "kitty", "cats"]
    
    if any(keyword in query_lower for keyword in dog_keywords):
        context["target_species"] = "dog"
    elif any(keyword in query_lower for keyword in cat_keywords):
        context["target_species"] = "cat"
    else:
        # Default to dog if ambiguous (more common queries)
        context["target_species"] = None
    
    return context


def get_generation_prompt_for_context(product_context: dict) -> str:
    """
    Select the appropriate generation prompt based on product context
    This is the router for product analysis generation
    
    Args:
        product_context: Dictionary with keys:
            - product_type: "treat" | "food" | "supplement" | "grooming" | "toy" | "litter" | None
            - target_species: "dog" | "cat" | None
            
    Returns:
        Specialized system instruction string for LLM
        
    Example:
        >>> context = {"product_type": "treat", "target_species": "dog"}
        >>> prompt = get_generation_prompt_for_context(context)
        >>> # Returns treat analysis specialized prompt
    """
    prompts = ProductAnalysisGenerationPrompts()
    
    product_type = product_context.get("product_type", "").lower() if product_context.get("product_type") else ""
    
    # Route based on product type
    if "treat" in product_type or "snack" in product_type:
        return prompts.get_treat_analysis_generation_prompt()
    
    if "food" in product_type:
        return prompts.get_food_analysis_generation_prompt()
    
    if "supplement" in product_type:
        return prompts.get_supplement_analysis_generation_prompt()
    
    if "grooming" in product_type or "shampoo" in product_type:
        return prompts.get_grooming_analysis_generation_prompt()
    
    if "toy" in product_type:
        return prompts.get_toy_analysis_generation_prompt()
    
    if "litter" in product_type:
        return prompts.get_litter_analysis_generation_prompt()
    
    # Default: Generic product analysis assistant
    return """You are a pet product analysis expert.
Analyze the product based on the protocols provided and give comprehensive recommendations.

Focus on:
- Ingredient quality and safety
- Nutritional value (if applicable)
- Allergen risks
- Price/value assessment
- Better alternatives if needed
- Which pets this product is best suited for

Provide clear, actionable analysis to help pet owners make informed decisions."""
