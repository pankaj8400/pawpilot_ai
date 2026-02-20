"""
test_workflow.py
Test the complete workflow to verify everything works
"""

import sys
import logging
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

logger = logging.getLogger(__name__)

print("="*70)
print("Testing Complete Workflow")
print("="*70)

# Step 1: Test state creation
print("\n[STEP 1] Testing State Creation...")
try:
    from src.workflow.state_definition import WorkFlowState, create_initial_state
    
    state = create_initial_state(
        query="What is a dog?",
        user_id="test_user",
        session_id="test_session"
    )
    print("✓ State created successfully")
    print(f"  - Query: {state.get('query')}")
    print(f"  - User ID: {state.get('user_id')}")
except Exception as e:
    print(f"✗ State creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Test workflow compilation
print("\n[STEP 2] Testing Workflow Compilation...")
try:
    from src.workflow.graph_builder import build_complete_workflow
    
    workflow = build_complete_workflow()
    print("✓ Workflow compiled successfully")
except Exception as e:
    print(f"✗ Workflow compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test individual nodes
print("\n[STEP 3] Testing Individual Nodes...")
try:
    from src.workflow.nodes import (
        input_processing_node,
        decision_router_node
    )
    
    # Create test state
    test_state = {
        "query": "What is machine learning?",
        "user_id": "user_1",
        "session_id": "sess_1",
        "error": [],
        "start_time": 0
    }
    
    # Test input processing node
    print("  Testing input_processing_node...")
    result = input_processing_node(test_state)
    print(f"    ✓ Input processing works")
    
    # Test decision router node
    print("  Testing decision_router_node...")
    result = decision_router_node(result)
    print(f"    ✓ Decision router works")
    print(f"    - use_rag: {result.get('use_rag')}")
    print(f"    - strategy: {result.get('strategy')}")
    
except Exception as e:
    print(f"✗ Node testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test workflow execution
print("\n[STEP 4] Testing Workflow Execution...")
try:
    # Create initial state
    initial_state = {
        "query": "What is a dog?",
        "user_id": "user_1",
        "session_id": "sess_1",
        "use_rag": True,
        "error": [],
        "start_time": 0,
        "retrieved_docs": [],
        "context": "",
        "model_to_use": "gpt-4-turbo",
        "strategy": "prompt_only",
        "messages": [],
        "final_prompt": "",
        "raw_response": "",
        "response_tokens": 0,
        "cost": 0.0,
        "validated_response": "",
        "citations": [],
        "confidence_score": 0.0,
        "end_time": 0.0,
        "inference_time": 0.0,
        "total_time": 0.0,
        "fallback_used": False,
        "final_output": ""
    }
    
    print("  Invoking workflow...")
    result = workflow.invoke(initial_state)
    print("✓ Workflow executed successfully")
    print(f"  - Query: {result.get('query')}")
    print(f"  - Strategy: {result.get('strategy')}")
    print(f"  - Errors: {result.get('error')}")
    
except Exception as e:
    print(f"✗ Workflow execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test with ChatbotPipeline
print("\n[STEP 5] Testing ChatbotPipeline...")
try:
    # Import the pipeline
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from main import ChatbotPipeline
    
    # Create pipeline
    bot = ChatbotPipeline(workflow)
    print("✓ ChatbotPipeline created")
    
    # Test message processing
    print("  Processing message...")
    response = bot.process_message("What is a dog?")
    print("✓ Message processed successfully")
    print(f"  Response: {response}")
    
except Exception as e:
    print(f"✗ ChatbotPipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour workflow is ready to use!")
print("Run: python workflow_pipeline.py")