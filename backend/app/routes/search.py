
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.services.search import SearchService
from app.database.images import db_get_images_by_ids

router = APIRouter()

@router.get("/", response_model=List[dict])
async def search_images(
    q: str = Query(..., min_length=1, description="Search query string"),
    limit: int = 50
):
    """
    Search images using natural language.
    """
    try:
        search_service = SearchService.get_instance()
        image_ids = search_service.search_text(q, limit)
        
        if not image_ids:
            return []

        # Fetch image details
        results = db_get_images_by_ids(image_ids)
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
