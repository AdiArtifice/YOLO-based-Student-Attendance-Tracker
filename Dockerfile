FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for OpenCV and friends (libGL provider on Debian)
RUN apt-get update \
	 && apt-get install -y --no-install-recommends \
		 libgl1 \
		 libglib2.0-0 \
	 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY frontend/ ./frontend
COPY .env ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build image
# docker build -t attendance-app .
# Run locally
# docker run --env-file .env -p 8000:8000 attendance-app
# Or with Compose
# docker-compose up --build
# Test
# visit http://localhost:8000 and http://localhost:8000/docs
