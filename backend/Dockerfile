# Not built/tested in this environment (no local Docker) — written to
# standard conventions and worth a `docker build` check before relying on
# it. See ../docs/deployment.md for the full deployment story, including
# why model artifacts and data are handled as a mounted volume here
# rather than baked into the image.

FROM python:3.11-slim

WORKDIR /app

# LightGBM's precompiled wheel is linked against libgomp (OpenMP) at
# import time, not just training time — python:3.11-slim doesn't ship it,
# so `import lightgbm` fails at container startup with "libgomp.so.1:
# cannot open shared object file" even though pip install succeeds
# cleanly. Found the hard way deploying Dockerfile.demo to Render.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install dependencies before copying the rest of the source so dependency
# installation is cached across builds that only change application code.
RUN pip install --no-cache-dir -e .

COPY app ./app

# Trained model artifacts (data/models/*.joblib) and feature/label parquet
# files are gitignored, machine-generated, and too large to bake into the
# image sensibly. Mount ../data at /data (see docs/deployment.md) rather
# than COPY-ing it in — this is the standard "code in the image, data in a
# volume" split, and it means updating served predictions is a data
# refresh + restart, not a rebuild. MARKET_DATA_ROOT tells app/data/paths.py
# to look there instead of computing a path relative to its own file
# location (which assumes a directory layout this image doesn't have).
ENV MARKET_DATA_ROOT=/data
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
