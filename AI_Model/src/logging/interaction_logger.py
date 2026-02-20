from workflow.state_definition import WorkFlowState
from typing import Dict
import logging
from datetime import datetime
import json 

logger = logging.getLogger(__name__)
class Node7InteractionLogger():
    """
    NODE 7: Log interactions and prepare for feedback
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        state = WorkFlowState()
    
    def log_interaction(self, state: Dict) -> Dict:
        """
        NODE 7: Log user interaction
        
        Stores complete interaction record for:
        - Analytics
        - Training data accumulation
        - Feedback collection
        """
        
        logger.info("=" * 70)
        logger.info("NODE 7: LOGGING & FEEDBACK COLLECTION")
        logger.info("=" * 70)
        
        try:
            # ============================================================
            # STEP 1: BUILD INTERACTION RECORD
            # ============================================================
            logger.info("STEP 1: Building interaction record...")
            
            total_time = state.get("total_time", 0)
            
            interaction_record = {
                "user_id": state.get("user_id", "anonymous"),
                "pet_id": state.get("pet_id"),
                "session_id": state.get("session_id"),
                "timestamp": datetime.now().isoformat(),
                
                # Input
                "query": state.get("query", ""),
                
                # Output
                "response": state.get("validated_response", ""),
                "citations": state.get("citations", []),
                
                # Metadata
                "module": state.get("prompt_module"),
                "model_used": state.get("model_used"),
                "rag_used": state.get("use_rag", False),
                "fine_tuned_model": "ft-" in state.get("model_used", ""),
                
                # Metrics
                "confidence_score": state.get("confidence_score", 0),
                "response_quality": state.get("response_quality", {}),
                
                # Timing (milliseconds)
                "timing": {
                    "rag_ms": state.get("rag_time", 0) * 1000,
                    "prompt_ms": state.get("prompt_time", 0) * 1000,
                    "inference_ms": state.get("inference_time", 0) * 1000,
                    "validation_ms": state.get("validation_time", 0) * 1000,
                    "total_ms": total_time * 1000
                },
                
                # Cost & Usage
                "cost_usd": state.get("cost", 0),
                "tokens_generated": state.get("response_tokens", 0),
                "tokens_prompt": state.get("prompt_tokens", 0),
                
                # Status
                "success": state.get("errors", []) == [],
                "errors": state.get("errors", [])
            }
            
            logger.info("✓ Record built")
            
            
            # ============================================================
            # STEP 2: SAVE TO DATABASE
            # ============================================================
            logger.info("STEP 2: Saving to database...")
            
            if self.db:
                self.db.save_interaction(interaction_record)
                logger.info("✓ Saved to database")
            else:
                logger.warning("No database connection, skipping DB save")
            
            
            # ============================================================
            # STEP 3: SAVE TO JSONL (for training accumulation)
            # ============================================================
            logger.info("STEP 3: Saving to training accumulation file...")
            
            try:
                with open("data/interactions/log.jsonl", "a") as f:
                    f.write(json.dumps(interaction_record) + "\n")
                logger.info("✓ Saved to JSONL")
            except Exception as e:
                logger.warning(f"Failed to save to JSONL: {str(e)}")
            
            
            # ============================================================
            # STEP 4: TRIGGER FEEDBACK UI
            # ============================================================
            logger.info("STEP 4: Triggering feedback collection...")
            
            # In production, this would trigger the UI to show feedback buttons
            state["feedback_id"] = f"feedback_{datetime.now().timestamp()}"
            logger.info(f"✓ Feedback UI triggered: {state['feedback_id']}")
            
            
            # ============================================================
            # STEP 5: UPDATE STATE
            # ============================================================
            state["interaction_logged"] = True
            state["interaction_id"] = interaction_record.get("timestamp")
            
            logger.info("=" * 70)
            logger.info("NODE 7 COMPLETE - Interaction logged")
            logger.info("=" * 70)
            
            return state
        
        except Exception as e:
            logger.error(f"Error in Node 7: {str(e)}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Logging error: {str(e)}"]
            return state
