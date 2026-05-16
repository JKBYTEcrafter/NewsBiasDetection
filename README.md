# NewsBiasDetection

A Hindi news bias detection project that combines model training, inference, and deployment-ready applications.

## Repository structure

- `deployment_app/`
  - Production-ready Python app using FastAPI and a trained classifier.
  - Includes `train.py`, `app/`, `model/`, `static/`, `Dockerfile`, and `requirements.txt`.
- `hf_space_clone/`
  - A Hugging Face Spaces clone of the deployment workspace, with the same app structure and deployment files.
- `models/`
  - Collection of model definitions and experiments, including logistic regression, SVM, Naive Bayes, LSTM, XGBoost, and ensemble variants.
- `Dataset For Research Paper.xlsx`
  - Source dataset used for training and evaluation.

## Key features

- Trains bias detection models for Hindi news text.
- Provides a FastAPI web application for prediction.
- Supports Docker-based deployment and Hugging Face Spaces deployment.
- Includes multiple model implementations for experimentation.

## How to use

### Train the model

1. Open a terminal in the `deployment_app/` folder.
2. Run:

```powershell
python train.py
```

This generates a trained model artifact under `deployment_app/model/`.

### Run the web application locally

1. In the `deployment_app/` folder, start the FastAPI server:

```powershell
cd deployment_app
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

2. Open `http://localhost:7860` in your browser.

## Deployment

- `deployment_app/` is ready for containerized deployment with its `Dockerfile`.
- `hf_space_clone/` is structured to mirror Hugging Face Spaces deployment.

## Notes

- The main project artifacts are under `deployment_app/`.
- The `models/` directory contains standalone model scripts for research and experimentation.
- Use `Dataset For Research Paper.xlsx` as the training dataset, or adapt `train.py` for your own data.
