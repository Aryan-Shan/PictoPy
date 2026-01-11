
import os
import ftfy
import regex as re
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPTokenizer
from app.logging.setup_logging import get_logger

logger = get_logger(__name__)

class CLIPModel:
    def __init__(self):
        self.image_model_path = os.path.join(
            os.path.dirname(__file__), "ONNX_Exports", "clip_image_model.onnx"
        )
        self.text_model_path = os.path.join(
            os.path.dirname(__file__), "ONNX_Exports", "clip_text_model.onnx"
        )
        self.tokenizer_path = os.path.join(
            os.path.dirname(__file__), "ONNX_Exports", "tokenizer"
        )
        
        self.tokenizer = None
        self.image_session = None
        self.text_session = None
        
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(self.tokenizer_path):
                 self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_path)
            else:
                # Fallback or error - expect download_models.py to have run
                logger.warning("CLIP Tokenizer not found locally. Attempting default 'openai/clip-vit-base-patch32'")
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            if os.path.exists(self.image_model_path):
                self.image_session = ort.InferenceSession(self.image_model_path)
            
            if os.path.exists(self.text_model_path):
                self.text_session = ort.InferenceSession(self.text_model_path)

        except Exception as e:
            logger.error(f"Error loading CLIP models: {e}")
            raise e

    def preprocess_text(self, text):
        # Basic cleanup
        text = ftfy.fix_text(text)
        text = text.lower().strip()
        return text

    def get_text_embedding(self, text):
        if not self.text_session or not self.tokenizer:
            logger.error("CLIP Text model not loaded")
            return None

        text = self.preprocess_text(text)
        inputs = self.tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="np")
        
        # ONNX Runtime expects specific input names, usually 'input_ids' and 'attention_mask'
        # Need to verify input names from the model, but standard export usually keeps them.
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        
        # The exported model usually takes input_ids and attention_mask
        # We might need to check session.get_inputs() to be sure of names
        input_feed = {
            self.text_session.get_inputs()[0].name: input_ids,
            self.text_session.get_inputs()[1].name: attention_mask
        }
        
        output = self.text_session.run(None, input_feed)[0]
        
        # Normalize
        embedding = output[0]
        norm = np.linalg.norm(embedding)
        return embedding / norm

    def get_image_embedding(self, image_path):
        if not self.image_session:
             logger.error("CLIP Image model not loaded")
             return None
        
        try:
            image = Image.open(image_path).convert("RGB")
            # Preprocessing for CLIP ViT-B/32
            # Resize to 224x224, Normalize
            input_data = self._preprocess_image(image)
            
            input_feed = {self.image_session.get_inputs()[0].name: input_data}
            output = self.image_session.run(None, input_feed)[0]
            
            embedding = output[0]
            norm = np.linalg.norm(embedding)
            return embedding / norm
        except Exception as e:
            logger.error(f"Error generating image embedding for {image_path}: {e}")
            return None

    def _preprocess_image(self, image):
        # Resize and Center Crop to 224
        w, h = image.size
        # Simple resize logic trying to match standard CLIP preprocessing
        # usually resize short edge to 224 then center crop.
        # For simplicity here: resize to 224x224 directly or robust approach
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        
        img_data = np.array(image).astype(np.float32)
        img_data = img_data / 255.0
        
        # Mean and Std for CLIP
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        
        img_data = (img_data - mean) / std
        
        # HWC -> CHW
        img_data = np.transpose(img_data, (2, 0, 1))
        
        # Add batch dimension: (1, 3, 224, 224)
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data
