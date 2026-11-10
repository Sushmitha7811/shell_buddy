FROM python:3.10-slim

WORKDIR /app

# 0. The base image ships an old pip that mishandles PyTorch's package index
RUN pip install --no-cache-dir --upgrade pip

# 1. CPU-only PyTorch (the default wheel bundles ~3 GB of GPU libraries we can't use)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Remaining dependencies, pinned
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# 3. A non-root user to run the app (created BEFORE copying, so it can own the files)
RUN useradd --create-home appuser

# 4. The model (rarely changes), then the code (changes often)
COPY --chown=appuser:appuser shell-buddy-merged/ ./shell-buddy-merged/
COPY --chown=appuser:appuser howto.py server.py ./

USER appuser

# 5. Docker checks /health to know if the app is actually working
HEALTHCHECK --interval=30s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"

EXPOSE 8765
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
