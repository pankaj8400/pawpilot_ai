"""
# Vision Prompt Templates - LangChain-based dynamic prompts
# Supports 4 approaches: Basic, Specialized, Few-Shot, Conditional
# """
from langchain_core.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate) 
from langchain.messages import SystemMessage, HumanMessage #to make the model recogineze the roles
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
        '    "injury_type": "...",\\n    "severity": "..."'
    """
    # Base fields (always include)
    base_fields = [
        '"injury_type": "..."',
        '"severity": "..."'
    ]
    
    # Standard adds location and classification
    standard_fields = base_fields + [
        '"body_location": "..."',
        '"species": "..."',
        '"caused_by": "..."'
    ]
    
    # Detailed adds clinical observations
    detailed_fields = standard_fields + [
        '"bleeding": "yes/no"',
        '"swelling": "yes/no"',
        '"contamination": "clean/dirty"',
        '"size_estimate": "small/medium/large"'
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

def create_injury_analysis_prompt(detail_level:str = "standard") -> ChatPromptTemplate:
    """
    Create a dynamic vision analysis prompt

    Args:
        detail_level: "basic" | "standard" | "detailed"
    
    Returns: ChatPromptTemplate ready for vision model
    Example:
        >>> prompt = create_injury_analysis_prompt("detailed")
        >>> messages = prompt.format_messages(fields="...", additional_instructions="...")
    """

    # TODO: Define system template (AI's role)
    system_template = """You are a medical image analysis assistant specializing in injury assessment.
    Your role: Analyze injury images and extract structured information for first aid guidance.
    Detail Level : {detail_level}
    Output Format: JSON only, no markdown, no explanations

    Be accurate and clinically relevant."""

    human_template = """Analyze this injury image carefully. Extract the following information:
1. **Injury Type**: What type of injury is this? (cut, burn, bite, scratch, bruise, etc.)
2. **Body Location**: Where on the body is the injury? (hand, arm, leg, face, paw, etc.)
3. **Severity**: How severe is the injury? (minor, moderate, severe, emergency)
4. **Species**: Who is injured? (human, dog, cat)
5. **Caused By**: If it's an animal-caused injury, what animal? (cat, dog, or N/A)
Return ONLY a JSON object with these keys:
{fields}

Additional Instructions: {additional_instructions}"""

    #TODO: Create the ChatPromptTemeplate
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    # Create default values for template variables
    #This makes the the template usable with .format_messages()
    chat_prompt = chat_prompt.partial(
        detail_level =detail_level,
        fields = get_json_fields_for_level(detail_level),
        additional_instructions="Provide accurate assessment based on visible evidence."
    )

    return chat_prompt

# =====================================================
# Backward Compatibility - Default Prompt
# =====================================================

# Create a default prompt for existing code that imports "prompt"
# This maintains compatibility with vision_integration.py

_default_template = create_injury_analysis_prompt(detail_level="standard")
_default_messages = _default_template.format_messages() # will fill thr dynamic variable details

# Extract the human message text (what the vision model actually sees)
prompt = _default_messages[1].content

# For advanced users who want the full template:
prompt_template = create_injury_analysis_prompt 

# =====================================================
# OPTION 2: Specialized Prompts (Different Scenarios)
# =====================================================

class VisionPromptTemplates:
    """
    Collection of specialized prompt templates for different injury scenarios
    Each method returns a ChatPromptTemplate optimized for that specific case
    """

    @staticmethod
    def get_human_injury_prompt() -> ChatPromptTemplate:
        """
        Specialized template for HUMAN injuries
        Focus: First aid priority, infection risk, medical attention timing
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a HUMAN injury. Critical focus areas:
- First aid priority actions
- Infection risk indicators  
- When to seek medical attention
- Emergency warning signs
- Tetanus risk assessment

Be conservative with severity - err on side of caution."""),
            ("human", """Analyze this human injury image.

Extract and assess:
1. Injury type and body location
2. Severity level (minor/moderate/severe/emergency)
3. Visible complications (bleeding, swelling, contamination, discoloration)
4. Urgency for medical care (immediate/within hours/routine/self-care)

Return JSON format with relevant medical guidance fields.""")
        ])

    @staticmethod
    def get_pet_injury_prompt() -> ChatPromptTemplate:
        """
        Specialized template for PET injuries (dogs/cats)
        Focus: Species identification, distress signs, vet care urgency
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a PET injury (dog or cat). Critical focus areas:
- Species identification (dog/cat)
- Pain/distress signs (hiding, not eating, limping, whining)
- Vet care urgency assessment
- Pet-safe first aid (NO human medications!)
- Object-caused injuries (metal, plastic, glass)

Be conservative with severity - err on side of caution."""),
            ("human", """Analyze this PET injury image.

Extract and assess:
1. Species (dog/cat) and injury type
2. Body location and caused by which object (e.g., metal, plastic toy, glass)
3. Severity level (minor/moderate/severe/emergency)
4. Visible complications (bleeding, swelling, limping)
5. Vet care timeframe (immediate/within hours/next day/monitor at home)

Return JSON format with pet-specific guidance fields.""")
        ])

    @staticmethod
    def get_bite_wound_prompt() -> ChatPromptTemplate:
        """
        Specialized for BITE WOUNDS (highest infection risk!)
        Focus: What animal, infection signs, rabies risk, puncture depth
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a BITE WOUND caused by an animal. CRITICAL focus areas:
- Animal identification (dog, cat, other)
- Infection risk (50-80% for cat bites, 20-30% for dog bites!)
- Rabies assessment (stray/unknown animal = HIGH RISK)
- Puncture depth (deep/moderate/shallow)
- Time sensitivity (bite wounds need care within 12-24 hours)

Be VERY conservative - bite wounds are high-risk injuries!"""),
            ("human", """Analyze this BITE WOUND injury image.

Extract and assess:
1. Animal that caused bite (dog/cat/other) and if appears domestic or stray
2. Body location and puncture depth estimate
3. Severity level (moderate/severe/emergency) - bites are rarely "minor"
4. Visible signs: bleeding, swelling, redness, pus
5. Urgency: URGENT care within 8-12 hours for most bites

Return JSON format with bite-specific risk assessment.""")
        ])


# =====================================================
# OPTION 3: Few-Shot Learning (Improved Accuracy)
# =====================================================

def create_fewshot_vision_prompt() -> ChatPromptTemplate:
    """
    Use few-shot examples to teach the AI proper output format

    Benefits:
    -More consistent JSON structure
    -Better handling of edge cases 
    -Higher accuracy on field values

    Return:
        ChatPromptTemplate with embedded examples
    """

    #Define example inputs and ideal output
    examples= [
        {
            "description": "Deep puncture marks on human forearm with swelling and redness",
            "output": '{"injury_type": "bite", "body_location": "forearm", "severity": "moderate", "species": "human", "caused_by": "cat"}'
        },
        {
            "description": "Red, blistered skin on human palm, no breaks in skin",
            "output": '{"injury_type": "burn", "body_location": "palm", "severity": "minor", "species": "human", "caused_by": "N/A"}'
        },
        {
            "description": "Dog paw pad bleeding, glass visible in wound",
            "output": '{"injury_type": "cut", "body_location": "paw_pad", "severity": "moderate", "species": "dog", "caused_by": "N/A"}'
        },
        {
            "description": "Superficial scratch marks on human face, barely bleeding",
            "output": '{"injury_type": "scratch", "body_location": "face", "severity": "minor", "species": "human", "caused_by": "cat"}'
        }
    ]
    #TODO: Create example prompt template
    #This defines how each exmaple is shown
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Image shows: {description}"),
        ("ai", "{output}")
    ])
    # This combines all examples together 
    from langchain.prompts import FewShotChatMessagePromptTemplate

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    #TODO: Create final prompt with examples + new request
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical image analysis assistant.
Learn from these example analyses, then analyze the new image following the EXACT same format."""),
        few_shot_prompt,  # Inserts all examples here
        ("human", "Now analyze this NEW injury image and provide JSON in the same format.")
    ])

    return final_prompt

# =====================================================
# OPTION 4: Conditional Prompt Selection (Smart Router)
# =====================================================

def get_prompt_for_context(injury_context: Optional[Dict]=None) -> ChatPromptTemplate:
    """
    Intelligently select the best prompt template based on context
    
    This is the "smart router" - it chooses the optimal template based on
    what you know about the injury before analysis.
    
    Args:
        injury_context: Dictionary with optional keys:
            - "suspected_type": "bite" | "burn" | "cut" | None
            - "species": "human" | "pet" | None  
            - "detail_level": "basic" | "standard" | "detailed"
            - "use_examples": True | False (use few-shot)
            
    Returns:
        The most appropriate ChatPromptTemplate
        
    Examples:
        >>> # Dog bite - use specialized bite template
        >>> prompt = get_prompt_for_context({"suspected_type": "bite"})
        
        >>> # Pet injury - use pet template
        >>> prompt = get_prompt_for_context({"species": "pet"})
        
        >>> # Need high accuracy - use few-shot
        >>> prompt = get_prompt_for_context({"use_examples": True})
        
        >>> # Unknown injury - use basic template
        >>> prompt = get_prompt_for_context()
    """
    # Default to empty context if none provided
    if injury_context is None:
        injury_context = {}
    
    # Create templates instance for Option 2
    templates = VisionPromptTemplates()
    
    # Priority 1: Use few-shot if requested (highest accuracy)
    if injury_context.get("use_examples"):
        return create_fewshot_vision_prompt()
    
    # Priority 2: Check for specialized scenarios (Option 2)
    suspected_type = injury_context.get("suspected_type", "").lower()
    
    # If it's a bite wound → Use bite-specific template
    if "bite" in suspected_type:
        return templates.get_bite_wound_prompt()
    
    # Priority 3: Check species (human vs pet)
    species = injury_context.get("species", "").lower()
    
    if "pet" in species or "dog" in species or "cat" in species:
        return templates.get_pet_injury_prompt()
    
    if "human" in species:
        return templates.get_human_injury_prompt()
    
    # Priority 4: Use Option 1 with specified detail level
    detail_level = injury_context.get("detail_level", "standard")
    return create_injury_analysis_prompt(detail_level)


# =====================================================
# OPTION 5: First Aid Generation Prompts (NEW)
# =====================================================

class FirstAidGenerationPrompts:
    """
    Specialized prompts for generating first aid instructions
    Mirrors the vision prompt specialization for consistent expertise
    
    Usage: Use same prompt strategy for both vision analysis and instruction generation
    """
    
    @staticmethod
    def get_human_injury_generation_prompt() -> str:
        """
        Specialized prompt for generating HUMAN injury first aid instructions
        
        Focus areas:
        - Immediate safety and stabilization
        - Infection prevention protocols
        - Medical attention timing (when to go to ER vs. urgent care vs. home care)
        - Emergency warning signs
        - Tetanus risk assessment
        
        Returns:
            System instruction string for LLM
        """
        return """You are a medical first aid expert specializing in HUMAN injuries.

CRITICAL FOCUS AREAS:
- Immediate safety actions and stabilization
- Infection prevention (cleaning, bandaging, monitoring)
- When to seek medical attention (emergency vs. urgent vs. routine vs. self-care)
- Emergency warning signs (arterial bleeding, shock, severe burns, head trauma)
- Tetanus risk assessment for penetrating wounds

INSTRUCTION STYLE:
- Be clear, direct, and action-oriented
- Start with most urgent actions first
- Use numbered steps for clarity
- Include specific timeframes (e.g., "seek care within 12 hours")
- Mention red flags that require immediate emergency care
- Be conservative with severity - err on side of caution

Based on the medical protocols provided, generate step-by-step first aid instructions."""

    @staticmethod
    def get_pet_injury_generation_prompt() -> str:
        """
        Specialized prompt for generating PET injury first aid instructions (dogs/cats)
        
        Focus areas:
        - Pet-safe first aid (NO human medications!)
        - Calming stressed/injured animals
        - Veterinary care urgency assessment
        - Home monitoring vs. emergency vet
        - Preventing self-injury (licking, scratching)
        
        Returns:
            System instruction string for LLM
        """
        return """You are a veterinary first aid expert specializing in PET injuries (dogs and cats).

CRITICAL FOCUS AREAS:
- Pet-safe first aid ONLY (NO human medications - many are toxic to pets!)
- Calming and restraining injured/stressed animals safely
- Veterinary care urgency (immediate emergency vet vs. within hours vs. next day vs. monitor at home)
- Home monitoring for worsening symptoms (won't eat, hiding, limping, whining)
- Preventing pet from licking/scratching wound (E-collar, bandaging)
- Species-specific considerations (cats hide pain, dogs may become aggressive when hurt)

INSTRUCTION STYLE:
- Clear, actionable steps for pet owners
- Safety first (for both pet and owner)
- Include signs that mean "go to vet NOW"
- Explain what's normal vs. concerning for recovery
- Mention monitoring duration (e.g., "monitor for 24 hours")
- Emphasize when professional vet care is mandatory

Based on the medical protocols provided, generate step-by-step pet first aid instructions."""

    @staticmethod
    def get_bite_wound_generation_prompt() -> str:
        """
        Specialized prompt for generating ANIMAL BITE WOUND first aid instructions
        
        Focus areas:
        - Extremely high infection risk (50-80% for cat bites!)
        - Rabies assessment and urgency
        - Thorough wound cleaning protocol
        - Antibiotic prophylaxis consideration
        - Medical care timing (within 8-12 hours critical)
        
        Returns:
            System instruction string for LLM
        """
        return """You are a medical expert specializing in ANIMAL BITE WOUND treatment.

CRITICAL FOCUS AREAS:
- HIGH INFECTION RISK (50-80% for cat bites, 20-30% for dog bites) - emphasize this!
- Rabies assessment urgency (stray/unknown animal = HIGH RISK, needs immediate care)
- Thorough wound cleaning protocol (15+ minutes of flushing with soap and water)
- Antibiotic prophylaxis (often needed for cat bites, deep punctures, hand/face bites)
- Medical care timing: URGENT - within 8-12 hours for most bites
- Infection signs to watch for (redness spreading, swelling, warmth, pus, red streaks, fever)

SPECIAL CONSIDERATIONS:
- Cat bites: Small punctures but VERY high infection risk (bacteria deep in tissue)
- Dog bites: More tearing damage, lower infection risk but more tissue damage
- Hand/face bites: Higher complication risk, need medical evaluation
- Deep punctures: Cannot clean deeply, high infection risk

INSTRUCTION STYLE:
- URGENT and serious tone - bite wounds are high-risk!
- Emphasize immediate thorough cleaning (most critical step)
- Clearly state "seek medical care within 8-12 hours"
- List infection warning signs explicitly
- Mention rabies risk assessment
- Be very conservative - when in doubt, recommend professional care

Based on the medical protocols provided, generate URGENT step-by-step bite wound first aid instructions."""

    @staticmethod
    def get_burn_injury_generation_prompt() -> str:
        """
        Specialized prompt for generating BURN injury first aid instructions
        
        Focus areas:
        - Burn degree assessment (1st, 2nd, 3rd)
        - Cooling protocol timing and method
        - Blister management (don't pop!)
        - Infection prevention
        - When burns need professional care
        
        Returns:
            System instruction string for LLM
        """
        return """You are a medical expert specializing in BURN injury treatment.

CRITICAL FOCUS AREAS:
- Burn degree classification (1st: superficial, 2nd: blistering, 3rd: deep tissue)
- Immediate cooling protocol (cool running water for 10-20 minutes, NOT ice)
- Blister management (DO NOT pop blisters - they protect against infection)
- Infection prevention (clean dressing, monitor for signs)
- When to seek care (2nd degree over large area, 3rd degree any size, face/hands/genitals, electrical/chemical burns)

SPECIAL CONSIDERATIONS:
- Size matters: burns larger than 3 inches need medical care
- Location matters: face, hands, feet, joints, genitals need medical care
- Electrical/chemical burns ALWAYS need emergency care
- Children and elderly: lower threshold for seeking care

INSTRUCTION STYLE:
- Clear cooling protocol first (stop the burning process)
- Emphasize what NOT to do (no ice, no butter, don't pop blisters)
- Specific criteria for seeking medical care
- Pain management suggestions (over-the-counter pain relief)
- Infection monitoring over 24-48 hours

Based on the medical protocols provided, generate step-by-step burn injury first aid instructions."""


# =====================================================
# OPTION 6: Text Query Context Detection (NEW)
# =====================================================

def detect_injury_context_from_text(query: str) -> dict:
    """
    Analyze text query to automatically determine injury context
    Used for text-only queries when no vision analysis is available
    
    This enables the same specialized prompting for text queries as image queries!
    
    Args:
        query: User's text description of injury
        
    Returns:
        Dictionary with injury context:
        {
            "suspected_type": "bite" | "burn" | "cut" | "scratch" | None,
            "species": "human" | "pet" | None,
            "detail_level": "standard"
        }
        
    Example:
        >>> detect_injury_context_from_text("my cat bit my hand")
        {"suspected_type": "bite", "species": "human", "detail_level": "standard"}
        
        >>> detect_injury_context_from_text("dog paw is bleeding from glass")
        {"suspected_type": "cut", "species": "pet", "detail_level": "standard"}
    """
    query_lower = query.lower()
    
    context = {
        "suspected_type": None,
        "species": None,
        "detail_level": "standard"
    }
    
    # Detect injury type (priority order: bite > burn > cut > scratch)
    if any(word in query_lower for word in ["bite", "bitten", "bit me", "animal bite"]):
        context["suspected_type"] = "bite"
    elif any(word in query_lower for word in ["burn", "burned", "burnt", "scalded", "scald"]):
        context["suspected_type"] = "burn"
    elif any(word in query_lower for word in ["cut", "laceration", "gash", "slash", "sliced"]):
        context["suspected_type"] = "cut"
    elif any(word in query_lower for word in ["scratch", "scratched", "claw", "clawed"]):
        context["suspected_type"] = "scratch"
    
    # Detect species
    # Pet indicators
    pet_keywords = ["my dog", "my cat", "my pet", "dog's", "cat's", "pet's", 
                    "paw", "tail", "fur", "puppy", "kitten"]
    
    # Human indicators (default if nothing detected)
    human_keywords = ["my hand", "my arm", "my leg", "my finger", "my face", 
                     "i was", "i got", "me ", "myself"]
    
    if any(keyword in query_lower for keyword in pet_keywords):
        context["species"] = "pet"
    elif any(keyword in query_lower for keyword in human_keywords):
        context["species"] = "human"
    else:
        # Default to human if ambiguous
        context["species"] = "human"
    
    return context


def get_generation_prompt_for_context(injury_context: dict) -> str:
    """
    Select the appropriate generation prompt based on injury context
    This is the router for first aid instruction generation
    
    Args:
        injury_context: Dictionary with keys:
            - suspected_type: "bite" | "burn" | "cut" | None
            - species: "human" | "pet" | None
            
    Returns:
        Specialized system instruction string for LLM
        
    Example:
        >>> context = {"suspected_type": "bite", "species": "human"}
        >>> prompt = get_generation_prompt_for_context(context)
        >>> # Returns bite wound specialized prompt
    """
    prompts = FirstAidGenerationPrompts()
    
    # Safely get values with None handling
    suspected_type = injury_context.get("suspected_type") or ""
    species = injury_context.get("species") or ""
    
    # Priority 1: Bite wounds (highest risk, most specialized)
    if suspected_type.lower() == "bite":
        return prompts.get_bite_wound_generation_prompt()
    
    # Priority 2: Burns (specific protocol needs)
    if suspected_type.lower() == "burn":
        return prompts.get_burn_injury_generation_prompt()
    
    # Priority 3: Check species (human vs pet)
    species_lower = species.lower()
    
    if "pet" in species_lower or "dog" in species_lower or "cat" in species_lower:
        return prompts.get_pet_injury_generation_prompt()
    
    if "human" in species_lower:
        return prompts.get_human_injury_generation_prompt()
    
    # Default: Generic medical assistant
    return """You are a medical first aid assistant.
Provide clear, step-by-step first aid instructions based on the protocols provided.
Be accurate, actionable, and prioritize safety."""

