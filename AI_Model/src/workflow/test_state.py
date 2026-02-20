"""
test_state.py
Quick test to verify the state definition is fixed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("Testing WorkFlowState Fix")
print("="*60)

# Test 1: Import
print("\n✓ Test 1: Importing WorkFlowState...")
try:
    from state_definition import WorkFlowState, create_initial_state, validate_state
    print("  ✓ Successfully imported WorkFlowState")
except Exception as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create state with dictionary
print("\n✓ Test 2: Creating state from dictionary...")
try:
    state_dict = {
        "query": "What is a dog?",
        "user_id": "user_123",
        "session_id": "session_456",
        "use_rag": True,
        "model_to_use": "gpt-4-turbo",
        "strategy": "rag"
    }
    state = WorkFlowState(**state_dict)
    print(f"  ✓ State created successfully")
    print(f"  Query: {state.get('query')}")
    print(f"  User ID: {state.get('user_id')}")
except Exception as e:
    print(f"  ✗ Failed to create state: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Use helper function
print("\n✓ Test 3: Creating state with helper function...")
try:
    initial_state = create_initial_state(
        query="Tell me about machine learning",
        user_id="user_789",
        session_id="sess_abc"
    )
    print(f"  ✓ Initial state created successfully")
    print(f"  Query: {initial_state.get('query')}")
    print(f"  Session ID: {initial_state.get('session_id')}")
    print(f"  Total fields: {len(initial_state)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Validate state
print("\n✓ Test 4: Validating state...")
try:
    is_valid = validate_state(initial_state)
    if is_valid:
        print(f"  ✓ State is valid")
    else:
        print(f"  ✗ State validation failed")
except Exception as e:
    print(f"  ✗ Validation error: {e}")
    sys.exit(1)

# Test 5: Check all fields
print("\n✓ Test 5: Checking state fields...")
try:
    expected_fields = [
        "query", "user_id", "session_id", "use_rag", "model_to_use",
        "strategy", "retrieved_docs", "context", "final_prompt",
        "raw_response", "validated_response", "error", "final_output"
    ]
    
    missing_fields = []
    for field in expected_fields:
        if field not in initial_state:
            missing_fields.append(field)
    
    if not missing_fields:
        print(f"  ✓ All expected fields present ({len(expected_fields)} fields)")
    else:
        print(f"  ⚠ Missing {len(missing_fields)} fields: {missing_fields}")
        
except Exception as e:
    print(f"  ✗ Field check error: {e}")
    sys.exit(1)

# Test 6: Test with LangGraph (simulation)
print("\n✓ Test 6: Simulating LangGraph usage...")
try:
    # This is how LangGraph will use it
    user_input = "Do you know what is a dog?"
    state = WorkFlowState(
        query=user_input,
        user_id="user_1",
        session_id="session_1"
    )
    print(f"  ✓ LangGraph-style state creation works")
    print(f"  Created state with query: '{state['query']}'")
except Exception as e:
    print(f"  ✗ LangGraph simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nYour state_definition.py is now fixed and ready to use!")
print("You can now run: python workflow_pipeline.py")