FROM python:3.9-slim-buster

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app" # Or ENV PYTHONPATH /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 5001
CMD ["python", "src/app.py"]