from typing import TypedDict
import numpy as np
class VisionWorkFlowState(TypedDict, total=False):
    """
    Class is used as standard definition for various states in the vision model.
    """
    query : str
    user_id : str
    session_id : str
    strategy : str
   
    image : np.ndarray
    model_to_use : str
    
    #Response
    predicted_class : str
    confidence_score : float
    
    #Lorence model 
    raw_model2_response : str

    #retrived docs
    retrieved_docs : dict

    #logging & metrics
    start_time : float
    end_time : float
    inference_time : float
    total_time : float

    #error handling
    error : list    

    #final output
    final_output : str


def create_initial_state(
    query: str = "",
    user_id: str = "",
    session_id: str = "",
    use_rag: bool = True,
    model_to_use: str = "diseases_classifier",
    strategy: str = "default"
) -> VisionWorkFlowState:
    """
    Create the initial state for the vision model workflow.
    
    Args:
        query: User's input query
        user_id: Unique identifier for the user
        session_id: Unique identifier for the session
        use_rag: Whether to use RAG pipeline
        model_to_use: The vision model to be used
        strategy: Strategy for processing   """
    import time
    import uuid
    
    if not session_id or session_id == "session_1":
        session_id = str(uuid.uuid4())[:8]
    
    return VisionWorkFlowState(
        # Input
        query=query,
        user_id=user_id,
        session_id=session_id,
        strategy=strategy,
        # Model Selection
        model_to_use=model_to_use,
        
        # RAG Pipeline
        retrieved_docs={},
        
        # Logging & Metrics
        start_time=time.time(),
        end_time=0.0,
        inference_time=0.0,
        total_time=0.0,
        
        # Error Handling
        error=[],
        
        # Final Output
        final_output=""
    )

