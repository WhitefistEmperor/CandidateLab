FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt
COPY . .
ENV HOST=0.0.0.0
ENV PORT=8502
ENV DATA_DIR=/app/instance
RUN useradd --create-home appuser && mkdir -p /app/instance && chown -R appuser:appuser /app
USER appuser
EXPOSE 8502
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8502') + '/health', timeout=3)"
CMD ["python", "app.py"]
