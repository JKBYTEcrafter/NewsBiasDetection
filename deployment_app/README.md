# Hindi News Bias Classifier Deployment Architecture

This directory (`deployment_app`) is a fully isolated production workspace constructed explicitly for your Python SentenceTransformer + Logistic Regression model. It is optimized to run seamlessly locally or to be uploaded directly to **Hugging Face Spaces**.

## Complete Workflow Structure

1. **`train.py`**: A specialized, lightweight script designed to run *locally*. It ingests your dataset, generates semantic embeddings, fits the cross-validated logic regression model precisely to your specifications, and serializes the `.pkl` artifact into the `/model/` folder.
2. **`app/main.py`**: A high-performance Python FastAPI engine constructed to accept user endpoints (`/predict`) securely.
3. **`static/`**: An elite vanilla HTML/JS frontend built natively into your backend without Node overhead, presenting deep analysis animations dynamically.
4. **`Dockerfile`**: Containerization mapping tailored specifically to run Hugging Face spaces on Port 7860 using PyTorch cache persistence.

---

## Instructions: Running Locally

### Step 1: Create Model Artifact
Before running the web app, you must generate the frozen weights from your dataset! Run this inside the `deployment_app` folder:
```powershell
python train.py
```
*(This looks for your research Excel directly in the parent directory! If you want to use a specific dataset path, execute `python train.py "C:\path\to\your\file.xlsx"`)*

### Step 2: Start Web Application
Once `model/lr_model.pkl` is successfully compiled, start FastAPI native server:
```powershell
cd deployment_app
uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Open your browser to `http://localhost:7860` to access the pristine User Interface.

---

## Instructions: Pushing to Hugging Face Spaces

Deploying this entire platform to the cloud is completely free and requires zero code changes. 

1. **Create Space**: Navigate to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create New Space**.
2. **Environment Name**: Enter a name (e.g., `Hindi-Bias-Detector`).
3. **Select Architecture**: Under "Select the Space SDK", pick **Docker** (and select "Blank").
4. **Hardware**: "Free Space" is perfectly fine. Create Space!
5. **Upload Files**: Head to your space's "Files" tab. Click "Add file" > "Upload files".
6. Upload *all* contents of this `deployment_app` folder, **crucially ensuring `model/lr_model.pkl` is uploaded!** Your directory on Hugging Face MUST look exactly like this:
   - `app/main.py`, `app/model.py`, `app/utils.py`
   - `model/lr_model.pkl`
   - `static/index.html`, `static/style.css`, `static/app.js`
   - `Dockerfile`
   - `requirements.txt`
7. Click "Commit".

Hugging Face will automatically read the `Dockerfile`, install the strict `requirements.txt`, setup dynamic caching for PyTorch semantic analysis, and host your premium web application automatically on the internet!
