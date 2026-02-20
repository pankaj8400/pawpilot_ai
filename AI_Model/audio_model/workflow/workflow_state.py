from typing_extensions import TypedDict
from typing import List, Dict
class WorkflowState(TypedDict):
    audio_file : List
    user_id : str
    session_id : str

    response : List
    
    retrieved_docs : Dict
    
    final_output : str

    start_time : float
    end_time : float
    inference_time : float

def create_audio_state(user_id: str, session_id: str, audio_file: list) -> WorkflowState:
    return WorkflowState(
        audio_file=audio_file,
        user_id=user_id,
        session_id=session_id,
        response=[],
        retrieved_docs={},
        final_output="",
        start_time=0.0,
        end_time=0.0,
        inference_time=0.0
    )