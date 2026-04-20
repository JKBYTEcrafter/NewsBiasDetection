# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory to /code
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .

# Install dependencies carefully to cache layer
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire app logic and the pre-computed model weights
COPY . .

# Create cache directory for SentenceTransformers to prevent re-downloads on restart
ENV SENTENCE_TRANSFORMERS_HOME=/code/.cache
RUN mkdir -p /code/.cache && chmod 777 /code/.cache

# Expose port required strictly by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI server securely
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
