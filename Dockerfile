FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install fastapi uvicorn python-multipart numpy yt-dlp
WORKDIR /app
COPY . /app
WORKDIR /app/backend
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","10000"]
