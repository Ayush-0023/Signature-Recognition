import os
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request
)

from fastapi.templating import (
    Jinja2Templates
)

from src.pipeline.prediction import (
    PredictionPipeline
)


# Create FastAPI Application
app = FastAPI(
    title="Signature Recognition API",
    description=(
        "API for classifying signatures as "
        "Forged or Original using ResNet-34."
    ),
    version="1.0.0"
)


# Configure HTML Templates
templates = Jinja2Templates(
    directory="templates"
)


# Initialize Prediction Pipeline
predictor = PredictionPipeline()


# Temporary Upload Directory
UPLOAD_DIR = "temp_uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# Home Page
@app.get("/")
def home(
    request: Request
):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# Health Check
@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# Prediction Endpoint
@app.post("/predict")
async def predict_signature(
    file: UploadFile = File(...)
):

    temp_file_path = None

    try:

        # Validate file type
        allowed_types = [
            "image/jpeg",
            "image/jpg",
            "image/png"
        ]

        if file.content_type not in allowed_types:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid file type. "
                    "Please upload a JPG, JPEG, or PNG image."
                )
            )


        # Create unique temporary filename
        file_extension = os.path.splitext(
            file.filename
        )[1]

        unique_filename = (
            f"{uuid.uuid4()}"
            f"{file_extension}"
        )

        temp_file_path = os.path.join(
            UPLOAD_DIR,
            unique_filename
        )


        # Save uploaded image temporarily
        with open(
            temp_file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Run Prediction
        result = predictor.predict(
            temp_file_path
        )


        # Return JSON
        return {
            "filename": file.filename,
            "prediction": result[
                "prediction"
            ],
            "confidence": result[
                "confidence"
            ]
        }


    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # Delete temporary image
        if (
            temp_file_path
            and os.path.exists(
                temp_file_path
            )
        ):

            os.remove(
                temp_file_path
            )


# Run Application
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )