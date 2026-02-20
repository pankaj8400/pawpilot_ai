import logging
from typing import Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DocumentSourceExtractor:
    """
    Extract source information from documents
    
    Handles multiple document formats:
    - RAG retrieved documents (with metadata)
    - PDF files
    - Web pages (with URLs)
    - Text files
    - JSON documents
    - Custom formatted documents
    """
    
    def __init__(self):
        """Initialize document source extractor"""
        self.source_cache = {}  # Cache extracted sources
    
    
    # ====================================================================
    # MAIN FUNCTION: EXTRACT SOURCE
    # ====================================================================
    
    def extract_source_from_doc(self, doc: Dict) -> str:
        """
        Extract source information from a document
        
        THIS IS THE MAIN FUNCTION CALLED IN NODE 6
        
        Args:
            doc: Document dictionary with various possible fields:
                {
                    "title": "...",
                    "source": "...",
                    "url": "...",
                    "file_path": "...",
                    "author": "...",
                    "content": "...",
                    "metadata": {...}
                }
        
        Returns:
            String with formatted source citation
            Examples:
            - "PawPilot AI - Skin Condition Guide"
            - "https://pawpilot.com/skin-conditions"
            - "vet_handbook.pdf"
            - "EmoDetect - Canine Emotion Detection System"
        
        Example usage in Node 6:
            for i, doc in enumerate(state.retrieved_documents, 1):
                source = extract_source_from_doc(doc)
                citations.append(f"[{i}] {source}")
        """
        
        # Check cache first
        doc_hash = self._hash_doc(doc)
        if doc_hash in self.source_cache:
            return self.source_cache[doc_hash]
        
        logger.debug(f"Extracting source from document...")
        
        # Try different extraction methods in priority order
        
        # METHOD 1: Explicit source field
        if "source" in doc and doc["source"]:
            source = self._format_source(doc["source"])
            self.source_cache[doc_hash] = source
            return source
        
        # METHOD 2: Title + Author
        if "title" in doc:
            source = self._extract_from_title_author(doc)
            self.source_cache[doc_hash] = source
            return source
        
        # METHOD 3: URL/Web source
        if "url" in doc and doc["url"]:
            source = self._extract_from_url(doc["url"])
            self.source_cache[doc_hash] = source
            return source
        
        # METHOD 4: File path
        if "file_path" in doc and doc["file_path"]:
            source = self._extract_from_file_path(doc["file_path"])
            self.source_cache[doc_hash] = source
            return source
        
        # METHOD 5: Metadata field
        if "metadata" in doc and isinstance(doc["metadata"], dict):
            source = self._extract_from_metadata(doc["metadata"])
            if source:
                self.source_cache[doc_hash] = source
                return source
        
        # METHOD 6: Extract from content (look for document title in first line)
        if "content" in doc and doc["content"]:
            source = self._extract_from_content(doc["content"])
            self.source_cache[doc_hash] = source
            return source
        
        # FALLBACK: Generic source
        fallback = "Unknown Source"
        self.source_cache[doc_hash] = fallback
        logger.warning(f"Could not extract source from doc, using fallback: {fallback}")
        return fallback
    
    
    # ====================================================================
    # METHOD 1: EXPLICIT SOURCE FIELD
    # ====================================================================
    
    def _format_source(self, source: str) -> str:
        """
        Format explicit source string
        
        Examples:
        Input: "PawPilot AI Knowledge Base"
        Output: "PawPilot AI Knowledge Base"
        
        Input: "https://example.com/article"
        Output: "example.com/article"
        """
        
        # If it's a URL, extract domain
        if source.startswith(("http://", "https://")):
            return self._extract_from_url(source)
        
        # Clean up and return
        source = source.strip()
        
        # Remove common prefixes
        if source.startswith("From: "):
            source = source[6:]
        if source.startswith("Source: "):
            source = source[8:]
        
        return source
    
    
    # ====================================================================
    # METHOD 2: TITLE + AUTHOR
    # ====================================================================
    
    def _extract_from_title_author(self, doc: Dict) -> str:
        """
        Extract source from title and author fields
        
        Examples:
        - "Skin Conditions Guide" by "Dr. Smith" → "Dr. Smith - Skin Conditions Guide"
        - "PawPilot AI Handbook" → "PawPilot AI Handbook"
        """
        
        title = doc.get("title", "").strip()
        author = doc.get("author", "").strip()
        
        if author and title:
            return f"{author} - {title}"
        elif title:
            return title
        else:
            return "Unknown Source"
    
    
    # ====================================================================
    # METHOD 3: URL/WEB SOURCE
    # ====================================================================
    
    def _extract_from_url(self, url: str) -> str:
        """
        Extract source from URL
        
        Examples:
        Input: "https://pawpilot.com/guides/skin-conditions"
        Output: "pawpilot.com/guides/skin-conditions"
        
        Input: "https://vethandbook.org/resources/emergency"
        Output: "vethandbook.org/resources/emergency"
        """
        
        try:
            parsed = urlparse(url)
            
            # Get domain
            domain = parsed.netloc.replace("www.", "")
            
            # Get path
            path = parsed.path.strip("/")
            
            # Combine
            if path:
                source = f"{domain}/{path}"
            else:
                source = domain
            
            # Clean up
            source = source.rstrip("/")
            
            logger.debug(f"Extracted URL source: {source}")
            return source
        
        except Exception as e:
            logger.warning(f"Failed to parse URL {url}: {str(e)}")
            return url
    
    
    # ====================================================================
    # METHOD 4: FILE PATH
    # ====================================================================
    
    def _extract_from_file_path(self, file_path: str) -> str:
        """
        Extract source from file path
        
        Examples:
        Input: "/data/documents/vet_handbook.pdf"
        Output: "vet_handbook.pdf"
        
        Input: "documents/PawPilot_Training_Guide.docx"
        Output: "PawPilot_Training_Guide.docx"
        
        Input: "RAG_Database/skin_conditions/dermatitis.txt"
        Output: "skin_conditions/dermatitis.txt"
        """
        
        try:
            path = Path(file_path)
            
            # Get filename with extension
            filename = path.name
            
            # Get parent directory if available
            parent = path.parent.name
            
            # Combine for more context
            if parent and parent not in (".", "data", "documents", "rag"):
                source = f"{parent}/{filename}"
            else:
                source = filename
            
            # Remove common prefixes
            source = source.replace("_", " ")
            
            logger.debug(f"Extracted file source: {source}")
            return source
        
        except Exception as e:
            logger.warning(f"Failed to parse file path {file_path}: {str(e)}")
            return Path(file_path).name
    
    
    # ====================================================================
    # METHOD 5: METADATA
    # ====================================================================
    
    def _extract_from_metadata(self, metadata: Dict) -> Optional[str]:
        """
        Extract source from metadata dictionary
        
        Looks for common metadata fields:
        - source, title, author, filename, url, document_name
        """
        
        # Priority order of metadata fields
        priority_fields = [
            "source",
            "title",
            "document_name",
            "filename",
            "author",
            "name"
        ]
        
        for field in priority_fields:
            if field in metadata and metadata[field]:
                value = metadata[field]
                
                # Handle author field specially
                if field == "author":
                    title = metadata.get("title", "")
                    if title:
                        return f"{value} - {title}"
                    else:
                        return value
                
                return str(value)
        
        return None
    
    
    # ====================================================================
    # METHOD 6: EXTRACT FROM CONTENT
    # ====================================================================
    
    def _extract_from_content(self, content: str) -> str:
        """
        Extract source from document content
        
        Looks for:
        - Title in first line (enclosed in markers)
        - Title pattern (e.g., "# Title" in markdown)
        - Document identifier
        
        Examples:
        Input: "===== PawPilot AI Knowledge Base ====="
               "The following information..."
        Output: "PawPilot AI Knowledge Base"
        
        Input: "# Canine Emotion Detection System (EmoDetect)"
               "This document explains..."
        Output: "Canine Emotion Detection System (EmoDetect)"
        """
        
        try:
            lines = content.split('\n')
            
            # Check first few lines for title patterns
            for i, line in enumerate(lines[:5]):
                line = line.strip()
                
                if not line:
                    continue
                
                # Pattern 1: Markdown header
                if line.startswith("# "):
                    title = line[2:].strip()
                    if title:
                        return title
                
                # Pattern 2: Enclosed in equals
                if line.startswith("=") and line.endswith("="):
                    title = line.strip("= ").strip()
                    if title:
                        return title
                
                # Pattern 3: Enclosed in dashes
                if line.startswith("-") and line.endswith("-"):
                    title = line.strip("- ").strip()
                    if title:
                        return title
                
                # Pattern 4: All caps with colons
                if line.isupper() and ":" in line:
                    title = line.split(":")[0].strip()
                    if title:
                        return title
                
                # Pattern 5: First substantial line (not empty, not too long)
                if len(line) > 10 and len(line) < 150:
                    return line[:100] + "..." if len(line) > 100 else line
            
            return "Extracted Document"
        
        except Exception as e:
            logger.warning(f"Failed to extract from content: {str(e)}")
            return "Document"
    
    
    # ====================================================================
    # HELPER METHODS
    # ====================================================================
    
    def _hash_doc(self, doc: Dict) -> str:
        """Create hash of document for caching"""
        
        try:
            # Use content or URL as cache key
            if "content" in doc:
                return str(hash(doc["content"][:100]))
            elif "url" in doc:
                return str(hash(doc["url"]))
            elif "title" in doc:
                return str(hash(doc["title"]))
            else:
                return str(hash(str(doc)))
        except:
            return str(id(doc))
    
    
    def clear_cache(self) -> None:
        """Clear source extraction cache"""
        self.source_cache.clear()
        logger.info("Source extraction cache cleared")


# ============================================================================
# INTEGRATION: Used in Node 6
# ============================================================================

def extract_source_from_doc(doc: Dict) -> str:
    """
    Standalone function version for use in Node 6
    
    Called when formatting citations
    
    Example usage in Node 6:
        for i, doc in enumerate(state.retrieved_documents, 1):
            source = extract_source_from_doc(doc)
            citations.append(f"[{i}] {source}")
    """
    
    extractor = DocumentSourceExtractor()
    return extractor.extract_source_from_doc(doc)

