from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import io
            detail=f"Conversion failed: {str(e)}"
        )
