import hashlib
import json
from typing import Optional, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PromptCache:
    """
    Cache prompt responses to avoid redundant API calls and reduce costs
    """
    
    def __init__(self, cache_file: str = "data/prompt_cache.json", ttl_hours: int = 24):
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("Creating new cache file")
            return {}
    
    def get_cache_key(self, prompt: str, model: str, pet_id: Optional[str] = None) -> str:
        """
        Generate cache key from prompt and model
        
        For PawPilot: Include pet_id if available (different pets = different caches)
        """
        
        key_text = f"{prompt}:{model}:{pet_id or 'general'}"
        return hashlib.md5(key_text.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, pet_id: Optional[str] = None) -> Optional[str]:
        """
        Retrieve cached result if exists and not expired
        
        Returns:
            Cached response or None if not found/expired
        """
        
        key = self.get_cache_key(prompt, model, pet_id)
        
        if key not in self.cache:
            return None
        
        cached_item = self.cache[key]
        cached_time = datetime.fromisoformat(cached_item["timestamp"])
        
        # Check if cache expired
        if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
            logger.info(f"Cache expired for key: {key}")
            del self.cache[key]
            return None
        
        logger.info(f"Cache HIT for key: {key}")
        return cached_item["response"]
    
    def set(self, prompt: str, model: str, response: str, pet_id: Optional[str] = None):
        """
        Cache a response
        
        Args:
            prompt: The prompt used
            model: The model used
            response: The generated response
            pet_id: Optional pet ID (for pet-specific caching)
        """
        
        key = self.get_cache_key(prompt, model, pet_id)
        
        self.cache[key] = {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "pet_id": pet_id
        }
        
        self._save_cache()
        logger.info(f"Cached response for key: {key}")
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def clear_cache(self, pet_id: Optional[str] = None):
        """Clear all cached results (optionally for specific pet)"""
        
        if pet_id:
            # Clear only this pet's cache
            keys_to_delete = [
                key for key, value in self.cache.items()
                if value.get("pet_id") == pet_id
            ]
            for key in keys_to_delete:
                del self.cache[key]
            logger.info(f"Cleared cache for pet: {pet_id}")
        else:
            # Clear entire cache
            self.cache = {}
            logger.info("Cleared entire cache")
        
        self._save_cache()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        
        total_items = len(self.cache)
        expired_items = 0
        
        for item in self.cache.values():
            cached_time = datetime.fromisoformat(item["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                expired_items += 1
        
        return {
            "total_items": total_items,
            "expired_items": expired_items,
            "active_items": total_items - expired_items,
            "cache_file": self.cache_file,
            "ttl_hours": self.ttl_hours
        }
