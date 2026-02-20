import sys
import time
from pathlib import Path
import traceback
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.workflow.state_definition import WorkFlowState
from src.utils.exceptions import CustomException
from src.prompt_engineering.prompts import PawPilotPromptBuilder
import logging  

logger = logging.getLogger(__name__)

# Try importing RAG - if it fails, continue with mock
try:
    from src.rag.rag_pipline import RAGPipeline
    RAG_AVAILABLE = True
except Exception as e:
    logger.warning(f"RAG not available: {e}")
    RAG_AVAILABLE = False



# ============================================================================
# NODE 1: Input Processing Node
# ============================================================================

def input_processing_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 1: Validate and Process user Inputs 
    
    Responsibilities:
    - Validate the user query and IDs
    - Extract MetaData
    - Initialize TimeStamp
    - Set default values
    """
    logger.info("NODE 1: Input Processing")
    
    try:
        state['to_use_model'] = True
        # Access state as dictionary
        query = state.get("query", "")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        
        # Validate query
        if not query or query.strip() == "":
            logger.warning("Invalid/empty query provided")
            state["error"] = ["Invalid Query"]
            state["strategy"] = "error"
            return state
        
        # Truncate long queries
        if len(query) > 2000:
            logger.warning(f"Query too long ({len(query)} chars), truncating...")
            state["error"] = ["Query too long"]
            state["query"] = query[:2000]
        
        # Initialize metadata
        state["start_time"] = time.time()
        if "error" not in state or state["error"] is None:
            state["error"] = []
        state["fallback_used"] = False
        
        logger.info(f'Processing query from user {user_id} in session {session_id}')
        logger.info(f'Query: {query[:50]}...')
        
        return state
        
    except Exception as e:
        logger.error(f"Error in input_processing_node: {e}")
        state["error"] = [f"Input processing error: {str(e)}"]
        state["strategy"] = "error"
        return state


# ============================================================================
# NODE 2: Decision Router Node
# ============================================================================

def decision_router_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 2: Decides which technique to be used based on state inputs

    Responsibilities:
    - Analyze the Query types
    - Decide on RAG usage
    - Select Appropriate Model 
    - Choose Execution Strategy
    """
    logger.info("NODE 2: Decision Router")
    
    try:
        query = state.get("query", "").lower()
        
        
        use_rag = should_use_rag(query)
        
        state["use_rag"] = use_rag
        logger.info(f"RAG decision: use_rag = {use_rag}")
        
        # Check if fine-tuned model is available (for now, using base model)
        ft_model = None  # Placeholder
        
        if ft_model and hasattr(ft_model, 'performance_score') and ft_model.performance_score >= 0.9:
            state["model_to_use"] = str(ft_model.id)
            logger.info(f'Using fine-tuned model: {ft_model.id}')
        else:
            state["model_to_use"] = 'gpt-4-turbo'
            logger.info('Using base model: gpt-4-turbo')
        
        # Determine Execution Strategy
        if use_rag:
            state["strategy"] = "rag"
        else:
            state["strategy"] = "prompt_only"
        
        logger.info(f"Strategy: {state['strategy']}")
        return state
        
    except Exception as e:
        logger.error(f"Error in decision_router_node: {e}")
        state["error"] = [f"Router error: {str(e)}"]
        state["strategy"] = "error"
        return state


# ============================================================================
# NODE 3: RAG Retrieval
# ============================================================================

def rag_retrieval_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 3: Retrieve relevant documents from vector store

    Responsibilities:
    - Connect to vector store
    - Perform similarity search
    - Update state with retrieved documents
    """
    logger.info("NODE 3: RAG Retrieval")
    
    try:
        use_rag = state.get("use_rag", False)
        
        if not use_rag:
            logger.info('RAG not required, skipping document retrieval')
            state["context"] = ""
            state["retrieved_docs"] = []
            return state
        
        if not RAG_AVAILABLE:
            logger.warning("RAG pipeline not available, skipping")
            state["context"] = ""
            state["retrieved_docs"] = []
            return state
        
        logger.info("Executing RAG pipeline...")
        rag_start = time.time()
        
        try:
            # Step 1: Call RAG system
            rag_system = RAGPipeline()
            
            # Step 2: Retrieve documents
            query = state.get("query", "")
            docs = rag_system.retriever(query=query, top_k=4)
            
            # Step 3: Update state
            state["retrieved_docs"] = docs if docs else []
            state["context"] = "\n---\n".join(doc['text'] for doc in docs) if docs else ""
            state["rag_time"] = float(time.time() - rag_start)
            logger.info(f'RAG retrieved {len(docs)} documents in {state["rag_time"]:.2f}s')
            return state
            
        except Exception as e:
            logger.error(f"RAG execution error: {e}")
            if "error" not in state or state["error"] is None:
                state["error"] = []
            state["error"].append(f"RAG error: {str(e)}")
            state["context"] = ""
            state["retrieved_docs"] = []
            state["use_rag"] = False  # Fallback
            return state
    
    except Exception as e:
        logger.error(f"Error in rag_retrieval_node: {e}")
        state["error"] = [f"RAG retrieval error: {str(e)}"]
        return state


# ============================================================================
# NODE 4: Prompt Engineering
# ============================================================================

def engineer_prompt_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 4: Prompt Engineering
    
    Builds optimized prompts using PawPilotPromptBuilder
    """
    logger.info("NODE 4: Prompt Engineering")
    
    try:
        if state.get('to_use_model', True):
            query = state.get("query", "")
            context = state.get("context", "")
            strategy = state.get("strategy", "prompt_only")
            
            # Initialize prompt builder
            prompt_builder = PawPilotPromptBuilder()
            
            # Check if this is a vision workflow (has predicted_class from vision model)
            predicted_class = state.get("predicted_class", "")
            confidence_score = state.get("confidence_score", 0.0)
            model_type = state.get("strategy", "default")
            
            # Build pet profile from state if available
            pet_profile = {
                "name": state.get("pet_name", "Unknown"),
                "species": state.get("pet_species", "Unknown"),
                "breed": state.get("pet_breed", "Unknown"),
                "age": state.get("pet_age", "Unknown"),
                "weight": state.get("pet_weight", "Unknown"),
                "allergies": state.get("pet_allergies", []),
                "medical_history": state.get("pet_medical_history", "None reported"),
            }
            # Determine which prompt to build based on workflow type
            if predicted_class:
                if predicted_class != "unknown":
                    try:
                        prompt = prompt_builder.build_vision_prompt(
                            model_type=model_type,
                            predicted_class=predicted_class,
                            confidence_score=confidence_score,
                            user_query=query,
                            rag_context=context,
                            pet_profile=pet_profile,
                        
                        )
                        state['prompt_template'] = prompt
                    except Exception as e:
                        print(traceback.format_exc())
            

                else:
                    # Default vision prompt
                    prompt = prompt_builder.build_vision_default_prompt(
                        predicted_class=predicted_class,
                        confidence_score=confidence_score,
                        user_query=query,
                        rag_context=context,
                        strategy=strategy
                    )
                    state["prompt_template"] = prompt
                
            elif context and strategy == "rag":
                # RAG-based workflow - use RAG-aware prompts
                logger.info("Building RAG-aware prompt")
                
                # Determine module type based on query content
                query_lower = query.lower()
                
                if any(kw in query_lower for kw in ["emergency", "urgent", "bleeding", "poison", "choking"]):
                    module = "emergency"
                    user_query_dict = {
                        "emergency_type": "general",
                        "symptoms": query
                    }
                elif any(kw in query_lower for kw in ["skin", "rash", "bump", "itch", "hair loss", "wound"]):
                    module = "skin_diagnosis"
                    user_query_dict = {
                        "symptom_description": query
                    }
                elif any(kw in query_lower for kw in ["food", "treat", "ingredient", "safe to eat", "toxic"]):
                    module = "product_safety"
                    user_query_dict = {
                        "name": "Unknown Product",
                        "type": "food",
                        "ingredients": query
                    }
                else:
                    # Fallback to simple RAG prompt
                    prompt = f"""Based on the following context, answer the user's query:
    
                            Context:    
                            {context}
                                
                            User Query: {query}
                                
                            Please provide a comprehensive and accurate answer based on the provided context."""
                    state["final_prompt"] = prompt
                    state["prompt_template"] = "rag_default"
                    logger.info(f"Prompt engineered using default RAG template ({len(prompt)} chars)")
                    return state
                
                try:
                    prompt = prompt_builder.build_rag_aware_prompt(
                        module=module,
                        user_query=user_query_dict,
                        pet_profile=pet_profile,
                        rag_retrieved_data=context
                    )
                    state["prompt_template"] = f"rag_{module}"
                except Exception as e:
                    logger.warning(f"Failed to build specialized prompt: {e}, using default")
                    prompt = f"""Based on the following context, answer the user's query:

                            Context:    
                            {context}
                                
                            User Query: {query}
                                
                            Please provide a comprehensive and accurate answer based on the provided context."""
                    state["prompt_template"] = "rag_default"      
        return state
        
    except Exception as e:
        logger.error(f"Error in engineer_prompt_node: {e}")
        if "error" not in state or state["error"] is None:
            state["error"] = []
        state["error"].append(f"Prompt engineering error: {str(e)}")
        # Fallback to basic prompt
        state["final_prompt"] = state.get("query", "")
        state["prompt_template"] = "fallback"
        return state


# ============================================================================
# NODE 5: Model Inference
# ============================================================================

def run_model_inference_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 5: Model Inference
    
    Calls LLM with optimized prompt
    """
    logger.info("NODE 5: Model Inference")
    
    try:
        if state.get('to_use_model', True):
            from AI_Model.src.models.model_inference import Node5ModelInference
            node = Node5ModelInference()
            result = node.run_inference(state)
            logger.info("Model inference completed")
            state.update(result) 


        return state    
        
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        # Fallback response
        state["raw_response"] = f"I received your query: {state.get('query', 'No query')}. However, I encountered an error processing it: {str(e)}"
        state["response_tokens"] = len(state["raw_response"].split())
        state["cost"] = 0.0
        if "error" not in state or state["error"] is None:
            state["error"] = []
        state["error"].append(f"Model inference error: {str(e)}")
        raise CustomException(e,sys)
    finally:
        return state
 

# ============================================================================
# NODE 6: Response Validation & Formatting
# ============================================================================

def validate_response_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 6: Validate Response
    
    Validates and formats the model response
    """
    logger.info("NODE 6: Response Validation")
    
    try:
        if state.get('to_use_model', True):
            from AI_Model.src.utils.reponse_validator import Node6ResponseValidator
            validator = Node6ResponseValidator()
            validated_result = validator.validate_response(dict(state))
            state.update(validated_result)
            logger.info("Response validation completed")
        return state
        
    except Exception as e:
        logger.error(f"Response validation error: {e}")
        # Fallback: use raw response as validated response
        state["validated_response"] = state.get("raw_response", "")
        state["citations"] = []
        state["confidence_score"] = 0.5
        raise CustomException(e, sys)
    finally:
        return state


# ============================================================================
# NODE 7: Logging & Feedback Collection
# ============================================================================

def log_interaction_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 7: Log Interaction
    
    Logs the interaction for monitoring
    """
    logger.info("NODE 7: Logging & Feedback")
    
    try:
        if state.get('to_use_model', True):
            from AI_Model.src.logging.interaction_logger import Node7InteractionLogger
            logger_node = Node7InteractionLogger()
            result = logger_node.log_interaction(dict(state))
            state.update(result)
            logger.info("Interaction logged")
        return state
        
    except Exception as e:
        logger.error(f"Logging error: {e}")
        # Fallback: just track time
        state["end_time"] = time.time()
        state["total_time"] = state["end_time"] - state.get("start_time", state["end_time"])
        logger.info(f"Total time: {state['total_time']:.2f}s")
        return state


# ============================================================================
# NODE 8: Fine-Tuning Trigger
# ============================================================================

def check_fine_tuning_trigger_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 8: Check Fine-Tuning Trigger
    
    Determines if fine-tuning should be triggered
    """
    logger.info("NODE 8: Fine-Tuning Check")
    
    try:
        if state.get('to_use_model', True):
            from AI_Model.src.fine_tuning.fine_tuner import check_fine_tuning_trigger
            result = check_fine_tuning_trigger(dict(state))
            state.update(result)
            logger.info("Fine-tuning check completed")
        return state
        
    except Exception as e:
        logger.error(f"Fine-tuning check error: {e}")
        # Fallback: just return state
        state["final_output"] = state.get("validated_response", state.get("raw_response", ""))
        return state

def should_use_rag(query) -> bool:
    NON_RAG_PREFIXES = (
        "what is",
        "who is",
        "define",
        "meaning of",
        "introduction to",
        "explain"
    )

    MEDICAL_KEYWORDS = (
        "symptom",
        "disease",
        "treatment",
        "fever",
        "vomiting",
        "diarrhea",
        "skin",
        "infection",
        "breathing",
        "poison"
    )

    if query.startswith(NON_RAG_PREFIXES):
        return False

    if any(word in query for word in MEDICAL_KEYWORDS):
        return True

    return False


