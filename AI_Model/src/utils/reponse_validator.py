import logging
from typing import Dict, List
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from utils.exceptions import CustomException
from utils.documents_parser import DocumentSourceExtractor
logger = logging.getLogger(__name__)

class Node6ResponseValidator:
    """
    Node 6: Response Validation & Formatting
     
    Validates ressponse quality, extracts citations, format for user

    """

    def __init__(self):
        self.source_extractor = DocumentSourceExtractor()

    def validate_response(self, state : Dict) -> Dict:
        """
        Node 6: Validate and Format response

        Input from Node 5:
        - raw_response: Direct model output
        - response_tokens: Token count
        - retrieved_documents : RAG context

        Output:
        - Validated_response: Formatted, safe response
        - Citations : List of source citations
        - confidence_score: 0-1 confidence
        - response_quality: Quality assessment dict
        """
        
        logger.info("=" * 70)
        logger.info("NODE 6: RESPONSE VALIDATION & FORMATTING") 
        logger.info("=" * 70)

        try:
            # Step 1: Check response quality
            logger.info('Step 1: Checking response quality....')

            quality_check = self._check_response_quality(
                response = state['raw_response'],
                module = state.get("prompt_module", "general_qa")
            )

            if quality_check['status'] == 'FAILED':
                logger.error(f"Response quality check failed: {quality_check['message']}")
                state['validated_response'] = 'I apologies, but I was uanable to generate a proper response. Please try again.'
                state['response_quality'] = quality_check
                state['confidence_score'] = 0.0
            logger.info(f'✅ Quality check: {quality_check["status"]}')

            # Step 2: Extract Citations From RAG

            logger.info('Step 2: Extracting citations from RAG context....')

            citations = self._extract_citations(
                retrieved_documents = state.get("retreived_documents", []),
                use_rag = state.get('use_rag', False)
            )

            logger.info(f'✅ Response formatted ({len(citations)} citations')

            # Step 3: Format Reponse

            logger.info('Step 3: Formatting Response.......')

            formatted_response = self._format_response(
                raw_response = state['raw_response'],
                module = state.get('prompt_module', 'general_qa'),
                citations = citations 
            )

            logger.info(f'✅ Reponse formatted {len(formatted_response)} chars')


            # Step 4: Calculate confidence score

            logger.info('Step 4: Calculating confidence score....')

            confidence_score = self._calculate_confidence(
                response = formatted_response,
                has_rag = state.get('use_rag', False),
                quality_status = quality_check['status']
            )

            logger.info(f'✅ Confidence Score: {confidence_score:.2%}')

            # STEP 5: VALIDATE SAFETY
            logger.info("STEP 5: Validating response safety...")
            
            safety_check = self._check_response_safety(formatted_response)
            
            if not safety_check["is_safe"]:
                logger.warning(f"Safety issue detected: {safety_check['issues']}")
                state["validated_response"] = "This response cannot be provided due to safety concerns. Please contact support."
                state["confidence_score"] = 0.0
                state["safety_check"] = safety_check
                return state
            
            logger.info(f"✅ Safety check passed")
            
            
            # STEP 6: ADD EMERGENCY NOTICE (if applicable)
            if state.get("prompt_module") == "emergency":
                formatted_response = self._add_emergency_notice(formatted_response)
                logger.info("✅ Emergency notice added")

            
            # STEP 7: UPDATE STATE

            logger.info("STEP 7: Updating state with validation results...")
            
            state["validated_response"] = formatted_response
            state["citations"] = citations
            state["confidence_score"] = confidence_score
            state["response_quality"] = quality_check
            state["safety_check"] = safety_check        
        except Exception as e:
            logger.error(f"Error in Node 6: {str(e)}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Validation error: {str(e)}"]
            state["validated_response"] = state.get("raw_response", "Error generating response")
            raise CustomException(e, sys)
        
        finally:
            return state
        


        # HELPER METHODS  

    def _check_response_quality(self, response: str, module: str) -> Dict:
        """Check if response meets quality standards"""
        
        checks = {
            "status": "OK",
            "message": "Response quality acceptable",
            "issues": []
        }
        
        # Check 1: Not empty
        if not response or len(response.strip()) < 10:
            checks["status"] = "FAILED"
            checks["message"] = "Response too short"
            return checks
        
        # Check 2: Length limits
        if len(response) > 10000:
            checks["issues"].append("Response very long")
            checks["status"] = "WARNING"
        
        # Module-specific checks
        if module == "emergency":
            if not any(f"{i}." in response for i in range(1, 5)):
                checks["issues"].append("Emergency response missing numbered steps")
                checks["status"] = "WARNING"
            
            if not any(word in response.lower() for word in ["urgent", "immediately", "emergency", "vet"]):
                checks["issues"].append("Missing urgency indicators")
                checks["status"] = "WARNING"
        
        elif module == "skin_diagnosis":
            if not any(word in response.lower() for word in ["severity", "urgent", "vet"]):
                checks["issues"].append("Missing medical guidance")
                checks["status"] = "WARNING"
        
        elif module == "product_safety":
            if not any(word in response.lower() for word in ["safe", "toxic", "recommend", "avoid"]):
                checks["issues"].append("Missing safety recommendation")
                checks["status"] = "WARNING"
        
        return checks
    
    
    def _extract_citations(self, retrieved_documents: List[Dict], use_rag: bool) -> List[str]:
        """Extract source citations from retrieved documents"""
        
        citations = []
        
        if not use_rag or not retrieved_documents:
            return citations
        
        for i, doc in enumerate(retrieved_documents, 1):
            try:
                source = self.source_extractor.extract_source_from_doc(doc)
                citations.append(f"[{i}] {source}")
            except Exception as e:
                logger.warning(f"Failed to extract source from doc {i}: {str(e)}")
                citations.append(f"[{i}] Source {i}")
        
        return citations
    
    
    def _format_response(self, raw_response: str, module: str, citations: List[str]) -> str:
        """Format response for user display"""
        
        # Start with raw response
        formatted = raw_response.strip()
        
        # Add citations if available
        if citations:
            formatted += "\n\n## Sources\n"
            formatted += "\n".join(citations)
        
        # Add footer based on module
        if module == "emergency":
            formatted += "\n\n---\n⚠️ **IMPORTANT:** If your pet's condition worsens or doesn't improve, contact your veterinarian immediately."
        
        elif module == "product_safety":
            formatted += "\n\n---\n📋 **Always consult your veterinarian** before introducing new foods or products to your pet."
        
        elif module == "skin_diagnosis":
            formatted += "\n\n---\n🐾 **This is not a diagnosis.** Please see a veterinarian for proper diagnosis and treatment."
        
        # Remove excessive whitespace
        formatted = re.sub(r'\n\n\n+', '\n\n', formatted)
        
        return formatted
    
    
    def _calculate_confidence(self, response: str, has_rag: bool, quality_status: str) -> float:
        """Calculate confidence score (0-1)"""
        
        score = 0.5  # Start at 50%
        
        # Boost for quality
        if quality_status == "OK":
            score += 0.2
        
        # Boost for RAG context
        if has_rag:
            score += 0.15
        
        # Boost for well-structured response
        if response.count("##") >= 2 or response.count("\n") > 10:
            score += 0.1
        
        # Boost for citations
        if "[" in response and "]" in response:
            score += 0.05
        
        # Cap at 1.0
        return min(score, 1.0)
    
    
    def _check_response_safety(self, response: str) -> Dict:
        """Check for harmful content"""
        
        harmful_patterns = [
            r"don't see a vet",
            r"ignore.*doctor",
            r"guaranteed cure",
            r"definitely.*diagnosis",
            r"100%.*safe"
        ]
        
        issues = []
        for pattern in harmful_patterns:
            if re.search(pattern, response.lower()):
                issues.append(f"Potential harmful content: {pattern}")
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues
        }
    
    
    def _add_emergency_notice(self, response: str) -> str:
        """Add prominent emergency notice"""
        
        notice = "🚨 **EMERGENCY SITUATION - IMMEDIATE ACTION REQUIRED** 🚨\n\n"
        return notice + response
