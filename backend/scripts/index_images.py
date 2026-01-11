
import sys
import os
import asyncio

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.images import db_get_all_images
from app.services.search import SearchService
from app.logging.setup_logging import get_logger

logger = get_logger(__name__)

def index_all_images():
    print("Fetching all images from database...")
    images = db_get_all_images()
    print(f"Found {len(images)} images.")
    
    search_service = SearchService.get_instance()
    
    count = 0
    for img in images:
        path = img['path']
        img_id = img['id']
        
        # Check if file exists
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
            
        try:
            success = search_service.add_image(img_id, path)
            if success:
                count += 1
                if count % 10 == 0:
                    print(f"Indexed {count}/{len(images)} images...")
        except Exception as e:
            print(f"Failed to index {path}: {e}")

    search_service.save_index()
    print(f"Indexing complete. Successfully indexed {count} images.")

if __name__ == "__main__":
    index_all_images()
