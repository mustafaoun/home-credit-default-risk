FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
# Generate models/ locally before building; raw data is never needed by the server.
COPY models/model.pkl ./models/model.pkl
ENV MODEL_PATH=/app/models/model.pkl
ENV DECISION_THRESHOLD=0.12
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
