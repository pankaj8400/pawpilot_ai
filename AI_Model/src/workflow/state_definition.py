from typing import List, Dict, Any
from sqlalchemy import Boolean
from typing_extensions import TypedDict


class WorkFlowState(TypedDict, total=False):
    """
    This is a State object that flows through the graph nodes.
    Each Node receives this State object as input, modifies it and passes it to the next node.
    
    All fields are optional (total=False) so you can pass partial state.
    """

    # ========== INPUT ==========
    query: str                              # User's original question
    user_id: str                            # Track who asked
    session_id: str                         # Track conversation session

    # ========== DECISION MAKING ==========
    use_rag: bool                           # Should we retrieve documents?
    model_to_use: str                       # "gpt-4-turbo" or "ft-xyz123"
    strategy: str                           # "rag", "prompt_only", "hybrid"

    # ========== RAG PIPELINE OUTPUTS ==========
    retrieved_docs: List[Dict[str, Any]]           # Documents from vector DB
    rag_time: float                         # Time taken by RAG
    context: str                            # Formatted context string

    # ========== PROMPT ENGINEERING PIPELINE ==========
    prompt_template: str                    # Which template to use
    few_shots_examples: List[Dict[str, Any]]  # Example Q&A pairs
    final_prompt: str                       # Complete formatted prompt

    # ========== MODEL INFERENCE OUTPUTS ==========
    raw_response: str                       # Raw response from LLM
    response_tokens: int                    # Number of tokens in response
    cost: float                             # Cost incurred for this inference

    # ========== VALIDATION AND FORMATTING ==========
    validated_response: str                 # Validated and formatted response
    citations: List[str]                    # Source citations
    confidence_score: float                 # How confident are we? 0-1

    # ========== LOGGING AND METRICS ==========
    start_time: float                       # When did the request start
    end_time: float                         # When did the request end
    inference_time: float                   # Time taken for model inference
    total_time: float                       # Total time for the workflow

    # ========== ERROR HANDLING ==========
    error: List[str]                        # Any error encountered
    fallback_used: bool                     # Did we fall back to base model?

    # ========== FINAL OUTPUT ==========
    final_output: str                       # The final output to return to user
    
    # ========== HELPER FIELDS ==========
    messages: List[Dict[str, Any]]          # Message history for LangGraph
    user_input: str                         # Alias for query
    response: str                           # Alias for final_output
    output: str                             # Alias for final_output

    predicted_class : str

    to_use_model : bool
# ============================================================================
# Helper function to create initial state
# ============================================================================

def create_initial_state(
    query: str,
    user_id: str = "user_1",
    session_id: str = "session_1",
    use_rag: bool = True,
    model_to_use: str = "gpt-4-turbo",
    strategy: str = "prompt_only"
) -> WorkFlowState:
    """
    Create an initial state for a new conversation
    
    Args:
        query: User's input query
        user_id: User identifier (default: "user_1")
        session_id: Session identifier (default: "session_1")
        use_rag: Whether to use RAG (default: True)
        model_to_use: Which model to use (default: "gpt-4-turbo")
        strategy: Execution strategy (default: "prompt_only")
        
    Returns:
        Initial WorkFlowState dictionary with all fields populated
    """
    import time
    import uuid
    
    if not session_id or session_id == "session_1":
        session_id = str(uuid.uuid4())[:8]
    
    return WorkFlowState(
        # Input
        query=query,
        user_id=user_id,
        session_id=session_id,
        user_input=query,
        
        # Decision Making
        use_rag=use_rag,
        model_to_use=model_to_use,
        strategy=strategy,
        
        # RAG Pipeline
        retrieved_docs=[],
        rag_time=0.0,
        context="",
        
        # Prompt Engineering
        prompt_template="default",
        few_shots_examples=[],
        final_prompt="",
        
        # Model Inference
        raw_response="",
        response_tokens=0,
        cost=0.0,
        
        # Validation
        validated_response="",
        citations=[],
        confidence_score=0.0,
        
        # Logging & Metrics
        start_time=time.time(),
        end_time=0.0,
        inference_time=0.0,
        total_time=0.0,
        
        # Error Handling
        error=[],
        fallback_used=False,
        
        # Final Output
        final_output="",
        response="",
        output="",
        
        # Helper Fields
        messages=[]
    )


# ============================================================================
# Type checking helper (optional)
# ============================================================================

def validate_state(state: WorkFlowState) -> bool:
    """
    Validate that the state has required fields
    
    Args:
        state: WorkFlowState to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["query", "user_id", "session_id"]
    
    for field in required_fields:
        if field not in state:
            print(f"❌ Missing required field: {field}")
            return False
    
    return True