
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from app.models.CLIP import CLIPModel
from app.logging.setup_logging import get_logger
from app.config.settings import DATABASE_PATH

logger = get_logger(__name__)

INDEX_PATH = os.path.join(os.path.dirname(DATABASE_PATH), "search_index.faiss")
ID_MAP_PATH = os.path.join(os.path.dirname(DATABASE_PATH), "search_ids.pkl")

class SearchService:
    _instance = None
    
    def __init__(self):
        self.clip_model = CLIPModel()
        self.dimension = 512  # CLIP ViT-B/32 output dimension
        self.index = None
        self.id_map = [] # Maps FAISS internal ID to Image ID (str)
        self.is_dirty = False
        
        self.load_index()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(ID_MAP_PATH, 'rb') as f:
                    self.id_map = pickle.load(f)
                self.last_loaded = os.path.getmtime(INDEX_PATH)
                logger.info(f"Loaded search index with {self.index.ntotal} items.")
            except Exception as e:
                logger.error(f"Error loading search index: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.id_map = []
                self.last_loaded = 0
        else:
            logger.info("Creating new search index.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.id_map = []
            self.last_loaded = 0

    def _check_and_reload(self):
        if not os.path.exists(INDEX_PATH):
            return
            
        try:
            mtime = os.path.getmtime(INDEX_PATH)
            if mtime > self.last_loaded:
                logger.info("Index file changed, reloading...")
                self.load_index()
        except Exception as e:
            logger.error(f"Error checking reload: {e}")

    def save_index(self):
        if not self.is_dirty:
            return
            
        try:
            faiss.write_index(self.index, INDEX_PATH)
            with open(ID_MAP_PATH, 'wb') as f:
                pickle.dump(self.id_map, f)
            self.is_dirty = False
            logger.info("Saved search index to disk.")
        except Exception as e:
            logger.error(f"Error saving search index: {e}")

    def add_image(self, image_id: str, image_path: str):
        embedding = self.clip_model.get_image_embedding(image_path)
        if embedding is None:
            return False
            
        # Reshape for FAISS (1, 512)
        embedding = embedding.reshape(1, -1).astype('float32')
        
        self.index.add(embedding)
        self.id_map.append(image_id)
        self.is_dirty = True
        return True

    def search_text(self, query_text: str, limit: int = 50) -> List[str]:
        """
        Search for images matching the text query.
        Returns a list of Image IDs.
        """
        self._check_and_reload()
        
        if not query_text:
            return []
            
        embedding = self.clip_model.get_text_embedding(query_text)
        if embedding is None:
            return []
            
        embedding = embedding.reshape(1, -1).astype('float32')
        
        # Search
        # k is limit
        D, I = self.index.search(embedding, limit)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.id_map):
                results.append(self.id_map[idx])
                
        return results
