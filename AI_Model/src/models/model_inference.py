import sys
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
import os
import json
import time
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from openai import OpenAI, RateLimitError, APIConnectionError
from AI_Model.src.models.model_factory import ModelFactory
from AI_Model.src.utils.exceptions import CustomException
from AI_Model.src.utils.metrics import MetricsTracker
from AI_Model.src.utils.exceptions import CustomException
logger = logging.getLogger(__name__)
from AI_Model.src.utils.web_search import web_search

class Node5ModelInference:
    """
    NODE 5: Model Inference for PawPilot AI
    
    Responsibilities:
    - Select appropriate model (base or fine-tuned)
    - Call LLM with engineered prompt
    - Handle streaming response
    - Track tokens, cost, latency
    - Handle errors and retries
    - Monitor rate limits
    """
    
    def __init__(self, api_key: Optional[str] = os.getenv('OPENAIAPI')):
        """
        Initialize model inference engine
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_factory = ModelFactory()
        self.metrics = MetricsTracker()
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def run_inference(self, state) -> Dict:
        """
        NODE 5: Execute model inference with tool use support
        
        Args:
            state: WorkflowState with final_prompt and model selection
        
        Updates state with:
            - raw_response: Complete model output
            - response_tokens: Tokens generated
            - inference_time: How long inference took
            - cost: API cost for this request
            - model_used: Which model was used
        """
        
        logger.info("=" * 70)
        logger.info("NODE 5: MODEL INFERENCE")
        logger.info("=" * 70)
        
        try:
            # ============================================================
            # STEP 1: SELECT MODEL (Base vs Fine-tuned)
            # ============================================================
            logger.info("STEP 1: Selecting model...")
            
            model_name = self._select_model(state)
            state["model_used"] = model_name
            logger.info(f"✓ Model selected: {model_name}")
            
            
            # ============================================================
            # STEP 2: PREPARE INFERENCE PARAMETERS
            # ============================================================
            logger.info("STEP 2: Preparing inference parameters...")
            
            inference_params = self._prepare_inference_params(state, model_name)
            logger.info(f"✓ Parameters prepared: temp={inference_params['temperature']}, max_tokens={inference_params['max_tokens']}")
            
            
            # ============================================================
            # STEP 3: TRACK START TIME
            # ============================================================
            inference_start = time.time()
            logger.info("STEP 3: Starting inference timer...")
            
            
            # ============================================================
            # STEP 4: CALL MODEL WITH RETRY LOGIC (WITH TOOL USE LOOP)
            # ============================================================
            logger.info("STEP 4: Calling LLM with tool support...")
            prompt = state.get("final_prompt", "")
            if len(prompt)<1:
                prompt = state.get('prompt_template', 'No prompt template found')
            response, messages_history = self._call_model_with_tools(
                prompt=prompt,
                model=model_name,
                **inference_params
            )
            
            # ============================================================
            # STEP 5: EXTRACT RESPONSE
            # ============================================================
            logger.info("STEP 5: Extracting response...")
            
            raw_response = response.choices[0].message.content
            
            # Fallback if content is still None (shouldn't happen now)
            if raw_response is None:
                raw_response = "Unable to generate response"
            
            response_tokens = response.usage.completion_tokens if response.usage and hasattr(response.usage, 'completion_tokens') else len(raw_response.split())
            prompt_tokens = response.usage.prompt_tokens if response.usage and hasattr(response.usage, 'prompt_tokens') else 0
            
            state["raw_response"] = raw_response
            state["response_tokens"] = response_tokens
            state["prompt_tokens"] = prompt_tokens
            
            logger.info(f"✓ Response extracted ({response_tokens} tokens)")
            
            
            # ============================================================
            # STEP 6: CALCULATE METRICS
            # ============================================================
            logger.info("STEP 6: Calculating metrics...")
            
            inference_time = time.time() - inference_start
            cost = self._calculate_cost(model_name, prompt_tokens, response_tokens)
            
            state["inference_time"] = inference_time
            state["cost"] = cost
            state["model_latency_ms"] = inference_time * 1000
            
            logger.info(f"✓ Inference time: {inference_time:.2f}s")
            logger.info(f"✓ Cost: ${cost:.4f}")
            
            
            # ============================================================
            # STEP 7: TRACK METRICS FOR MONITORING
            # ============================================================
            logger.info("STEP 7: Tracking metrics...")
            
            self.metrics.record_inference(
                model=model_name,
                tokens=response_tokens,
                latency=inference_time,
                cost=cost,
                module=state.get("prompt_module", "unknown")
            )
            
            logger.info("✓ Metrics recorded")
            
            
            # ============================================================
            # STEP 8: ADD INFERENCE METADATA
            # ============================================================
            logger.info("STEP 8: Adding inference metadata...")
            
            state["inference_metadata"] = {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
                "inference_time_seconds": inference_time,
                "latency_ms": inference_time * 1000,
                "cost_usd": cost,
                "timestamp": datetime.now().isoformat(),
                "temperature": inference_params["temperature"],
                "max_tokens": inference_params["max_tokens"]
            }
            
            logger.info("✓ Metadata added")
            
            
            # ============================================================
            # STEP 9: CHECK RESPONSE QUALITY
            # ============================================================
            logger.info("STEP 9: Checking response quality...")
            
            quality_check = self._check_response_quality(
                response=str(raw_response),
                module_type=state.get("prompt_module", "general_qa")
            )
            
            state["response_quality"] = quality_check
            logger.info(f"✓ Quality check: {quality_check['status']}")
            
            if quality_check["status"] == "WARNING":
                logger.warning(f"Quality warning: {quality_check['message']}")
            
            
            logger.info("=" * 70)
            logger.info("NODE 5 COMPLETE - Ready for validation")
            logger.info("=" * 70)
            
            return state
        
        except Exception as e:
            logger.error(f"Unexpected error in Node 5: {str(e)}", exc_info=True)
            state.get("errors", []).append(f"Inference error: {str(e)}")
            raise CustomException(e,sys)
    
    
    # ====================================================================
    # HELPER METHODS
    # ====================================================================
    
    def _select_model(self, state: Dict)->str:
        """
        Select which model to use: fine-tuned or base
        
        Priority:
        1. If fine-tuned model exists and performing well → use it
        2. Otherwise → use base model
        """
        
        try:            
            # Check if we have an active fine-tuned model
            fine_tuned_model = None #self.model_factory.get_active_fine_tuned_model()
            
            if fine_tuned_model and fine_tuned_model.get("performance_score", 0) >= 0.85:
                logger.info(f"Using fine-tuned model: {fine_tuned_model['id']}")
                return fine_tuned_model["id"]
            
            # Fall back to base model
            base_model = self.model_factory.get_base_model()
            logger.info(f"Using base model: {base_model}")
            return base_model
        
        except Exception as e:
            logger.warning(f"Error selecting model, using default: {str(e)}")
            return "gpt-4-turbo"
    
    
    def _prepare_inference_params(self, state: Dict, model: str) -> Dict:
        """
        Prepare parameters for model inference
        
        Adjusts temperature, max_tokens, etc based on module type
        """
        
        module_type = state.get("prompt_module", "general_qa")
        
        # Module-specific parameters
        param_config = {
            "emergency": {
                "temperature": 0.3,  # Lower for consistency in emergencies
                "max_tokens": 300,   # Keep emergency responses concise
                "top_p": 0.9
            },
            "skin_diagnosis": {
                "temperature": 0.5,  # Balanced for medical accuracy
                "max_tokens": 500,
                "top_p": 0.95
            },
            "emotion_detection": {
                "temperature": 0.6,  # Slightly higher for nuance
                "max_tokens": 400,
                "top_p": 0.95
            },
            "product_safety": {
                "temperature": 0.4,  # Lower for accuracy
                "max_tokens": 450,
                "top_p": 0.9
            },
            "behavior": {
                "temperature": 0.7,  # Higher for creativity in training plans
                "max_tokens": 600,
                "top_p": 0.95
            }
        }
        # Get config for this module, or use defaults
        config = param_config.get(module_type, {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.95
        })
        
        return {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "top_p": config["top_p"],
            "stream": False  # For simplicity (can be enabled for streaming)
        }
    
    def _get_tools(self) -> list:
        """
        Configured all the available tools e.g., WebSearch and else(upcoming)
        """

        tools = [
                    {
                        "type" : "function",
                        "function" : {
                            "name" : "web_search",
                            "description" : "Search the web for pet health information, product recalls, and vet advice",
                            "parameters" : {
                                "type" : "object",
                                "properties" :{
                                    "query" :{
                                        "type" : "string",
                                        "description" : "The search query"
                                    }
                                },
                                "required" : ["query"]
                            }
                        }
                    }
                ]

        return tools
        
    
    def _call_model_with_retry(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool = False
    ):
        """
        Call OpenAI model with retry logic for resilience
        
        Retries on rate limits and connection errors
        """
        tools = self._get_tools()
        try:
            base_model = self.model_factory.get_base_model()
            client = OpenAI(api_key=os.getenv('OPENAIAPI'))
            response = client.chat.completions.create(
                model=base_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )
            return response
            
        except Exception as e:
            logger.error(f"Error during model call: {str(e)}", exc_info=True)
    
    
    def _call_model_with_tools(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool = False,
        max_tool_iterations: int = 3
    ):
        """
        Call OpenAI model with tool use support (agentic loop)
        
        Handles:
        1. Initial model call
        2. If model requests tool use → execute tool
        3. Feed tool result back to model
        4. Continue until model generates text response
        
        Args:
            prompt: Initial user prompt
            model: Model name to use
            max_tool_iterations: Max times to call tools before giving up
            
        Returns:
            Tuple of (final_response, messages_history)
        """
        tools = self._get_tools()
        client = OpenAI(api_key=os.getenv('OPENAIAPI'))
        base_model = self.model_factory.get_base_model()
        
        # Initialize messages (use Any to avoid strict type checking)
        messages: list[Any] = [{"role": "user", "content": prompt}]
        
        iteration = 0
        while iteration < max_tool_iterations:
            logger.info(f"Agentic Loop - Iteration {iteration + 1}/{max_tool_iterations}")
            iteration += 1
            
            # Call model
            response = client.chat.completions.create(
                model=base_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )
            
            # Check if model generated text content
            if response.choices[0].message.content is not None:
                logger.info(f"✓ Model generated text response (iteration {iteration})")
                return response, messages
            
            # Check if model wants to use tools
            if response.choices[0].message.tool_calls:
                logger.info(f"Model requesting tool use: {len(response.choices[0].message.tool_calls)} tool call(s)")
                
                # Add assistant's tool call to messages (omit content when None)
                messages.append({
                    "role": "assistant",
                    "tool_calls": response.choices[0].message.tool_calls
                })
                
                # Execute each tool call and collect results
                for tool_call in response.choices[0].message.tool_calls:
                    # Cast to Any to avoid type checking issues with tool_call structure
                    tool_call_any: Any = tool_call
                    
                    # Extract tool name and arguments safely
                    if hasattr(tool_call_any, 'function'):
                        tool_name = tool_call_any.function.name
                        tool_args = json.loads(tool_call_any.function.arguments)
                    else:
                        # Fallback for different tool_call structure
                        tool_name = getattr(tool_call_any, 'name', 'unknown')
                        arguments_str = getattr(tool_call_any, 'arguments', '{}')
                        tool_args = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    # Execute the tool
                    if tool_name == "web_search":
                        try:
                            query = tool_args.get("query", "")
                            # Handle LangChain BaseTool or regular callable
                            if hasattr(web_search, 'invoke'):
                                tool_result = web_search.invoke({"query": query})
                            elif callable(web_search):
                                tool_result = web_search(query)
                            else:
                                tool_result = "web_search tool not properly configured"
                            logger.info(f"✓ Web search returned: {len(str(tool_result))} characters")
                        except Exception as e:
                            tool_result = f"Error executing web search: {str(e)}"
                            logger.error(f"✗ Web search failed: {str(e)}")
                    else:
                        tool_result = f"Unknown tool: {tool_name}"
                        logger.warning(f"Unknown tool requested: {tool_name}")
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                
                # Loop continues to call model again with tool results
                logger.info("Tool results added to conversation, calling model again...")
            else:
                # No text content and no tool calls = something went wrong
                logger.warning(f"Unexpected state: no content and no tool calls")
                # Create a fallback response
                return response, messages
        
        # Max iterations reached
        logger.warning(f"Max tool iterations ({max_tool_iterations}) reached without getting text response")
        return response, messages
    
    
    def _calculate_cost(self, model: str, prompt_tokens: int, response_tokens: int) -> float:
        """
        Calculate API cost for this inference
        
        Pricing as of 2024 (update as needed)
        """
        
        pricing = {
            "gpt-4-turbo": {
                "input": 0.01 / 1000,      # $0.01 per 1K tokens
                "output": 0.03 / 1000      # $0.03 per 1K tokens
            },
            "gpt-4": {
                "input": 0.03 / 1000,
                "output": 0.06 / 1000
            },
            "gpt-3.5-turbo": {
                "input": 0.0005 / 1000,
                "output": 0.0015 / 1000
            },
            "ft-gpt-3.5-turbo": {
                "input": 0.003 / 1000,     # Fine-tuned pricing
                "output": 0.004 / 1000
            }
        }
        
        # Get pricing for model, default to base pricing
        model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
        
        cost = (prompt_tokens * model_pricing["input"]) + (response_tokens * model_pricing["output"])
        
        logger.info(f"Cost calculation: {prompt_tokens} input + {response_tokens} output = ${cost:.6f}")
        
        return cost
    
    
    def _check_response_quality(self, response: str, module_type: str) -> Dict:
        """
        Check if response meets quality standards for its module
        
        Returns: {status: "OK" | "WARNING" | "FAILED", message: str}
        """
        
        checks = {
            "status": "OK",
            "message": "Response quality is acceptable",
            "warnings": []
        }
        
        # Check 1: Response not empty
        if not response or len(response.strip()) < 10:
            checks["status"] = "FAILED"
            checks["message"] = "Response too short or empty"
            return checks
        
        # Module-specific checks
        if module_type == "emergency":
            # Emergency responses MUST have numbered steps
            if not any(f"{i}." in response for i in range(1, 10)):
                checks["status"] = "WARNING"
                checks["warnings"].append("Emergency response missing numbered steps")
            
            # Must mention urgency
            if not any(word in response.lower() for word in ["urgent", "immediately", "critical", "call vet"]):
                checks["status"] = "WARNING"
                checks["warnings"].append("Emergency response lacks urgency language")
        
        elif module_type == "skin_diagnosis":
            # Diagnosis should mention severity
            if not any(word in response.lower() for word in ["severity", "urgent", "high", "medium", "low"]):
                checks["warnings"].append("Missing severity assessment")
            
            # Should mention vet
            if "vet" not in response.lower():
                checks["status"] = "WARNING"
                checks["warnings"].append("Should mention veterinarian")
        
        elif module_type == "product_safety":
            # Product safety should have clear recommendation
            if not any(word in response.lower() for word in ["safe", "toxic", "recommend", "avoid", "unsafe"]):
                checks["status"] = "WARNING"
                checks["warnings"].append("Missing safety recommendation")
        
        elif module_type == "emotion_detection":
            # Should identify specific emotion
            emotions = ["fear", "anxiety", "happiness", "aggression", "curiosity", "trust", "stress", "submission", "playfulness"]
            if not any(emotion in response.lower() for emotion in emotions):
                checks["warnings"].append("May not have identified specific emotion")
        
        # Check for toxic patterns
        if self._contains_harmful_content(response):
            checks["status"] = "FAILED"
            checks["message"] = "Response contains harmful content"
            return checks
        
        # Check for hallucinations (very long lists, repetition)
        if response.count("\n") > 50:
            checks["warnings"].append("Response very long - possible hallucination")
        
        if checks["warnings"]:
            checks["status"] = "WARNING"
            checks["message"] = "; ".join(checks["warnings"])
        
        return checks
    
    
    def _contains_harmful_content(self, response: str) -> bool:
        """Check if response contains harmful medical advice or dangerous information"""
        
        harmful_phrases = [
            "always do this",
            "guaranteed cure",
            "don't see a vet",
            "ignore the doctor"
        ]
        
        return any(phrase.lower() in response.lower() for phrase in harmful_phrases)

if __name__ == "__main__":
    # Example usage
    node = Node5ModelInference()
    test_state = {
        "final_prompt": "What should I do if my dog is having a seizure?",
        "prompt_module": "emergency"
    }
    result_state = node.run_inference(test_state)
    print(result_state["raw_response"])