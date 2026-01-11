
import os
import shutil
from huggingface_hub import hf_hub_download
from transformers import CLIPTokenizer

def download_clip_onnx():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "app", "models", "ONNX_Exports")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Downloading models to {model_dir}...")
    
    repo_id = "Xenova/clip-vit-base-patch32"
    
    # Download Image Encoder
    print("Downloading Image Encoder...")
    try:
        image_model_path = hf_hub_download(repo_id=repo_id, filename="vision_model.onnx")
    except:
        print("vision_model.onnx not found, trying onnx/vision_model.onnx")
        image_model_path = hf_hub_download(repo_id=repo_id, filename="onnx/vision_model.onnx")
        
    # Move/Rename
    target_image_path = os.path.join(model_dir, "clip_image_model.onnx")
    shutil.copy(image_model_path, target_image_path)
    
    # Download Text Encoder
    print("Downloading Text Encoder...")
    try:
        text_model_path = hf_hub_download(repo_id=repo_id, filename="text_model.onnx")
    except:
        print("text_model.onnx not found, trying onnx/text_model.onnx")
        text_model_path = hf_hub_download(repo_id=repo_id, filename="onnx/text_model.onnx")

    target_text_path = os.path.join(model_dir, "clip_text_model.onnx")
    shutil.copy(text_model_path, target_text_path)
    
    # Download Tokenizer
    print("Downloading Tokenizer...")
    # We can just load and save the tokenizer using transformers
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    
    print("Download complete.")

if __name__ == "__main__":
    download_clip_onnx()
