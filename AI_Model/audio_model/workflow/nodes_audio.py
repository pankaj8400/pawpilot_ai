import sys 
from AI_Model.src.utils.exceptions import CustomException
from AI_Model.audio_model.workflow.workflow_state import WorkflowState
from AI_Model.audio_model.model.emotion_detection import predict_emotion
import time
def input_processing_node(state : WorkflowState) -> WorkflowState:
    try:
        if len(state["audio_file"]) == 0:
            raise ValueError("No audio file provided in the state.")
        state['start_time'] = time.time()
        return state
    except Exception as e:
        raise CustomException(e, sys)
    
def emotion_detection_node(state : WorkflowState) -> WorkflowState:
    try:
        audio_files = state["audio_file"]
        emotion_results = predict_emotion(audio_files)
        state["response"] = emotion_results[0]
        return state
    except Exception as e:
        raise CustomException(e, sys)
    
def retrieval_node(state : WorkflowState) -> WorkflowState:
    try:
        from AI_Model.audio_model.workflow.retreiver import retrieve_docs
        response = state.get("response", {})
        docs = retrieve_docs(query = str(response))
        state['retrieved_docs'] = docs if isinstance(docs, dict) else {"documents": docs}
        return state
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    state = WorkflowState({
        "audio_file": ["AI_Model/audio_model/data/dog_barking_13.wav"],
        "user_id": "default_user",
        "session_id": "default_session",
        "response": [],
        "final_output": "",
        "start_time": 0.0,
        "end_time": 0.0,
        "inference_time": 0.0,
        "retrieved_docs": {}
    })
    state = emotion_detection_node(state)
    print(state.get("response"))
