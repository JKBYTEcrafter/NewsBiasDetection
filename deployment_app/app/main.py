import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from app.model import MLPipeline
    from app.utils import clean_text
except ImportError:
    from model import MLPipeline
    from utils import clean_text

# Initialize App
app = FastAPI(title="NewsBiasDetector", version="1.0")

# ── Global exception handler — always return JSON, never plain text ──────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": f"Internal server error: {str(exc)}"}
    )

# Request Schema
class TextRequest(BaseModel):
    text: str

# Lazy-loaded model singleton
runner = None

@app.on_event("startup")
async def startup_event():
    global runner
    try:
        print("[STARTUP] Loading ML pipeline...")
        runner = MLPipeline()
        print("[STARTUP] ML pipeline loaded successfully.")
    except Exception as e:
        print(f"[STARTUP WARNING] Model could not load on startup: {e}")
        traceback.print_exc()

@app.post("/predict")
async def predict_api(body: TextRequest):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    global runner

    # Lazy load if startup failed
    if runner is None:
        try:
            print("[PREDICT] Lazy-loading ML pipeline...")
            runner = MLPipeline()
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "detail": f"Model not ready: {str(e)}. Ensure lr_model.pkl exists."
                }
            )

    # Clean text exactly as during training
    cleaned = clean_text(body.text)

    # Run prediction with full error capture
    try:
        result = runner.predict(cleaned)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Prediction failed: {str(e)}"}
        )

    return {
        "success": True,
        "input_text": body.text,
        **result
    }

# ── Health check endpoint ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": runner is not None}

# ── Mount static frontend ────────────────────────────────────────────────────
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
