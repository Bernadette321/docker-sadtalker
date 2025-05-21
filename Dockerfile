# STEP 1: Base Image - REPLACE with RunPod recommended PyTorch image if available
FROM nvcr.io/nvidia/pytorch:23.10-py3

# STEP 2: Set Environment Variables (Optional but good practice)
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PORT=8000

WORKDIR $APP_HOME

# STEP 3: Install System Dependencies
# Minimal dependencies - ffmpeg is likely needed.
# Others might be needed depending on the exact base image and application specifics.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# STEP 4: Copy application dependency manifest
COPY requirements.txt .

# STEP 5: Install Python Dependencies
# requirements.txt should be curated:
# - PyTorch/Torchaudio should NOT be listed (use from base image)
# - OpenCV might be needed explicitly (e.g., opencv-python-headless) if not well-provided by base.
RUN pip install --no-cache-dir -r requirements.txt

# STEP 6: Copy application code and models into the image
# This includes your 'app' directory and 'app/sadtalker/SadTalker_source' etc.
COPY . .
# Ensure all necessary model files and subdirectories are copied.
# If 'SadTalker_source' is very large, consider alternatives like downloading at runtime if RunPod supports it easily.

# STEP 7: Expose the application port
EXPOSE $PORT

# STEP 8: Define the entrypoint or command to run the application
# Using uvicorn directly. Ensure app.main:app is correct.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$PORT"]