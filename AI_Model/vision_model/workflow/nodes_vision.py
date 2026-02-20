import sys
from AI_Model.vision_model.workflow.state_definition_vision import VisionWorkFlowState
from time import time 
from AI_Model.vision_model.rag_vision.retriever_vision import retrieve_docs
from AI_Model.vision_model.model.image_detect_model import call_nvdia 
from AI_Model.src.utils.exceptions import CustomException
import logging

logger = logging.getLogger(__name__)
ROUTER_INSTANCE = None
SKIP_SECOND_MODEL_STRATEGIES = ("emotion-detection", "full-body-scan")



def input_processing_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Process user input from the state dictionary.
    
    Args:
        state: Current workflow state as a VisionWorkFlowState.
    """
    logger.info('Processing user input...')

    try:
        query = state.get("query", "")
        image = state.get("image", None)
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        if image is not None:
            state['image'] = image
        state["query"] = query
        logger.info(f'Processing query from user {user_id} in session {session_id}')

        if not query or query.strip() == "":
            logger.warning("Invalid/empty query provided")
            state['strategy'] = 'all_models'
            return state
        
        if len(query) > 2000:
            logger.warning(f"Query too long ({len(query)} chars), truncating...")
            state["error"] = ["Query too long"]
            state["query"] = query[:2000]
        
        # Initialize metadata
        state["start_time"] = time()
        
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    

def decision_router_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Decide on the model strategy based on the state.
    
    Args:
        state: Current workflow state as a VisionWorkFlowState.
    """
    logger.info('Deciding on model strategy...')

    try:
        from AI_Model.vision_model.utils.keyword_extractor import QueryRouter
        global ROUTER_INSTANCE
        if ROUTER_INSTANCE is None:
            ROUTER_INSTANCE = QueryRouter()
        router = ROUTER_INSTANCE
        query = state.get("query", "")
        result = router.route_query(query)
        state["strategy"] = result.primary_strategy.value if result.primary_strategy is not None else "unknown"
        state['confidence_score'] = result.primary_confidence if result.primary_confidence is not None else 0.0
        return state

    except Exception as e: 
        raise CustomException(e, sys)

def model_call_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Call the appropriate model based on the strategy in the state.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Calling the appropriate model...')

    try:
        strategy = state.get("strategy", "")
        image = state.get("image")
        
        if strategy is None:
            strategy = ""
        
        # Check if image is provided
        if image is None:
            logger.error("No image provided in state")
            state["predicted_class"] = "unknown"
            state["confidence_score"] = 0.0
            if "error" not in state or state.get("error") is None:
                state["error"] = []
            state["error"].append("No image provided")
            return state
        
        if strategy == "skin-and-health-diagnostic":
            from AI_Model.vision_model.model.diseases_model_prediction import predict, load_model
            model, preprocess, classifier, id2label, device = load_model()
            results = predict(image, preprocess, model, classifier, id2label, device)
            # Handle different return types
            if results is None:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
            elif isinstance(results, str):
                # If predict returns a string directly
                state["predicted_class"] = results
                state["confidence_score"] = 1.0
            elif isinstance(results, list) and len(results) > 0:
                # If predict returns a list of dicts
                if isinstance(results, dict):
                    state["predicted_class"] = results.get("label", "unknown")
                    state["confidence_score"] = results.get("confidence", 0.0)
                else:
                    # List of strings
                    state["predicted_class"] = str(results[0])
                    state["confidence_score"] = 1.0
            elif isinstance(results, dict):
                # If predict returns a single dict
                state["predicted_class"] = results.get("label", "error")
                state["confidence_score"] = results.get("confidence", 0.0)
            else:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
            
        elif strategy == "toys-safety-detection":
            from AI_Model.vision_model.model.toy_model_prediction import predict_toy, load_model_toy
            model, preprocess, classifier, id2label, device = load_model_toy()
            results = predict_toy(image, preprocess, model, classifier, id2label, device)
            
            # Handle different return types
            if results is None:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
            elif isinstance(results, str):
                # If predict returns a string directly
                state["predicted_class"] = results
                state["confidence_score"] = 1.0
            elif isinstance(results, list) and len(results) > 0:
                # If predict returns a list of dicts
                if isinstance(results, dict):
                    state["predicted_class"] = results.get("label", "unknown")
                    state["confidence_score"] = results.get("confidence", 0.0)
                else:
                    # List of strings
                    state["predicted_class"] = str(results[0])
                    state["confidence_score"] = 1.0
            elif isinstance(results, dict):
                # If predict returns a single dict
                state["predicted_class"] = results.get("label", "unknown")
                state["confidence_score"] = results.get("confidence", 0.0)
            else:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
        elif strategy == "emotion-detection":
            from AI_Model.vision_model.model.emotion_detection import chatbot_emotion_detection
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = chatbot_emotion_detection(user_query, images)
            state["final_output"] = str(reply)
            state["predicted_class"] = "emotion_detected"
            state["confidence_score"] = 1.0  # Assuming full confidence for text response
        
        elif strategy == "injury-assistance":
            from AI_Model.vision_model.model.injury_assistance import chatbot_injury_assistance
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = chatbot_injury_assistance(user_query, images)
            state['final_output'] = str(reply)
            state['confidence_score'] = 0.85
            state['predicted_class'] = "injury_analyzed"

        elif strategy == "pet-food-image-analysis":
            from AI_Model.vision_model.model.pet_food_image_analysis import chatbot_food_analyzer
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = chatbot_food_analyzer(user_query, images)
            state['final_output'] = str(reply)
            state['predicted_class'] = "food_analyzed"
            state['confidence_score'] = 0.9
        elif strategy == "full-body-scan":
            from AI_Model.vision_model.model.full_body_scan import chatbot_full_body_scan
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = chatbot_full_body_scan(user_query, images)
            state['final_output'] = str(reply)
            state['predicted_class'] = "full_body_scan_analyzed"
            state['confidence_score'] = 0.9

        elif strategy == "packaged-product-scanner":
            from AI_Model.vision_model.model.packaged_product_scanner import process_food_image
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = process_food_image(images)
            state['retrieved_docs'] = reply
            state['predicted_class'] = "packaged_product_analyzed"
            state['confidence_score'] = 0.9

        elif strategy == "home-enviroment-safety-scan":
            #from AI_Model.vision_model.model.home_enviroment_safety_scan import chatbot_home_safety_scan
            user_query = state.get("query", "")
            #images = [image]
            #reply = chatbot_home_safety_scan(user_query, images)
            #state['final_output'] = str(reply)
            #state['predicted_class'] = "home_safety_scan_analyzed"
            #state['confidence_score'] = 0.9
        
        elif strategy == "parasite-detection":
            from AI_Model.vision_model.model.parasites_detection import predict_parasites
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = predict_parasites(images)
            state['predicted_class'] = str(reply[0])
            confidence = reply[1]
            state['confidence_score'] = float(confidence) if isinstance(confidence, (int, float, str)) else 0.0

        elif strategy == "poop-vomit-detection":
            from AI_Model.vision_model.model.poop_vomit_detection import predict_poop_vomit
            user_query = state.get("query", "")
            images = image if isinstance(image, list) else [image]
            reply = predict_poop_vomit(images)
            state['predicted_class'] = reply['label']
            confidence = reply['confidence']
            state['confidence_score'] = float(confidence) if isinstance(confidence, (int, float, str)) else 0.04
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            state["predicted_class"] = "unknown"
            state["confidence_score"] = 0.0
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    

def second_model_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Optionally call a second model based on confidence score.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Evaluating need for second model call...')

    try:
        if state.get("strategy") not in SKIP_SECOND_MODEL_STRATEGIES:
            response = call_nvdia(state.get('image'), prompt=state.get('query', '')+" Predicted class : " + state.get('predicted_class', ''))  
            state['raw_model2_response'] = response.get('choices', [{}])[0].get('message', {}).get('content', '')      
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    

def retrieval_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Placeholder retrieval node for vision model workflow.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Retrieval node - no operation for vision model.')

    try:
        if state.get("strategy") not in SKIP_SECOND_MODEL_STRATEGIES or state.get("strategy") != "packaged-product-scanner":
            # Currently no retrieval logic for vision mode
            class_name = state.get("predicted_class", "unknown")
            strategy = state.get("strategy", "default")

            if strategy == 'skin-and-health-diagnostic':
                host_name = 'https://dog-disease-6i6jnuf.svc.aped-4627-b74a.pinecone.io'
            elif strategy == 'pet-food-image-analysis':
                host_name = 'https://pet-food-image-analysis-6i6jnuf.svc.aped-4627-b74a.pinecone.io'
            elif strategy == 'parasite-detection':
                host_name = 'https://parasite-detection-6i6jnuf.svc.aped-4627-b74a.pinecone.io'
            elif strategy == 'toys-safety-detection':
                host_name = "https://toy-detection-6i6jnuf.svc.aped-4627-b74a.pinecone.io"
            elif strategy == 'poop-vomit-detection':
                host_name = "https://poop-vomit-detection-6i6jnuf.svc.aped-4627-b74a.pinecone.io"
            docs = retrieve_docs(class_name, host_name)
            # Convert list to dict format if needed
            if strategy == 'pet-food-image-analysis':
                state['retrieved_docs'] = docs.get('metadata', {}) if isinstance(docs, dict) else {"documents": docs}
            else:
                if isinstance(docs, dict):
                    state["retrieved_docs"] = docs
                elif isinstance(docs, list):
                    state["retrieved_docs"] = {"documents": docs}
                else:
                    state["retrieved_docs"] = {"documents": []}
        print(state.get('retrieved_docs'))     
        state['end_time'] = time()
        start_time = state.get('start_time', 0)
        state['inference_time'] = state['end_time'] - start_time
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    
