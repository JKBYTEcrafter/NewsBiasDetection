import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

try:
    from app.model import MLPipeline
    from app.utils import clean_text
except ImportError:
    from model import MLPipeline
    from utils import clean_text

# Initialize App
app = FastAPI(title="Hindi News Bias Detector", version="1.0")

# Request Schema
class TextRequest(BaseModel):
    text: str

# Keep model unloaded at module level to prevent import crash, load lazily on first hook or explicit trigger
runner = None

@app.on_event("startup")
async def startup_event():
    global runner
    # Will fail if train.py hasn't been run locally to generate lr_model.pkl
    try:
        runner = MLPipeline()
    except Exception as e:
        print(f"Warning: Model could not be booted on startup: {e}")

@app.post("/predict")
async def predict_api(body: TextRequest):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    global runner
    if runner is None:
        try:
            runner = MLPipeline()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Model artifact missing. Run train.py first.")

    # Apply identical text cleaning from training
    cleaned = clean_text(body.text)
    
    # Run prediction
    result = runner.predict(cleaned)
    
    return {
        "success": True,
        "input_text": body.text,
        **result
    }

# Mount frontend precisely
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
