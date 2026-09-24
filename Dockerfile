# WallExtractor annotation editor (FastAPI + static editor). CPU only, ~400 MB RAM.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 EDITOR_DATA=/data
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-editor.txt ./
RUN pip install -r requirements-editor.txt

COPY wallextractor ./wallextractor
COPY editor ./editor
COPY scripts ./scripts

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["uvicorn", "editor.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
