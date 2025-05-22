# 1. Base Image: Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# 2. Set Working Directory: Set the working directory in the container
WORKDIR /app

# 3. Copy NLTK data download script (if needed for stopwords, etc. during build or runtime)
# For now, our preprocess.py downloads nltk data if not found.
# If we wanted to pre-download during build:
# COPY src/preprocess.py /app/src/preprocess.py
# RUN python -c "from src.preprocess import preprocess_text; preprocess_text('test')"
# This would trigger the nltk.download('stopwords') if not present in the base image.
# For simplicity, we'll let it download at runtime if needed, or assume it's there.

# 4. Copy Requirements File: Copy the requirements file into the container
COPY requirements.txt .

# 5. Install Dependencies: Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code: Copy the local 'src' directory to '/app/src' in the container
COPY src/ /app/src/

# 7. Copy Models: Copy the local 'models' directory to '/app/models' in the container
# This assumes you've run 'dvc pull' or 'dvc repro' locally before building the image
# so that 'models/sentiment_model.joblib' and 'models/tfidf_vectorizer.joblib' exist.
COPY models/ /app/models/

# 8. Expose Port: Make port 5001 available to the world outside this container
# This is the port our Flask app runs on (as defined in src/app.py)
EXPOSE 5001

# 9. Environment Variable (Optional but good practice)
# Tells Flask to run in production mode (though our app.py currently forces debug=True)
# We should ideally make debug mode configurable via an env var in app.py
ENV FLASK_ENV=production 
ENV FLASK_APP=src/app.py 
# ENV FLASK_DEBUG=0 # To override debug mode if app.py respected it

# 10. Define a health check (optional but good for orchestrators)
# This checks if the app is responding on the home endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/ || exit 1

# 11. Run app.py when the container launches
# CMD ["python", "src/app.py"]
# Using gunicorn for a more production-ready server (even for local Docker runs)
# Install gunicorn first by adding it to requirements.txt and rebuilding.
# For now, stick with python src/app.py for simplicity of this step.
# If using gunicorn, the command would be something like:
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "src.app:app"]
# Let's stick to the simpler direct python execution for now.
CMD ["python", "src/app.py"]