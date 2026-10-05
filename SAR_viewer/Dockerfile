# PROTAC SAR matrix — container image.
#
# Build:  docker build -t sar-matrix .
# Run:    docker run -d --name sar --restart unless-stopped \
#             --env-file sar.env -v sar-data:/data \
#             -p 127.0.0.1:8050:8050 sar-matrix
#
# sar.env (chmod 600, never committed) holds cddVaultId=... and cddAPIToken=...
# — see .env.example. Publishing on 127.0.0.1 keeps the app off the network;
# in AWS the token comes from Secrets Manager instead and only the load
# balancer reaches port 8050.
#
# The image contains code only — no assay data, no credentials. The app
# downloads its copy of the CDD data into /data on first use.
FROM python:3.12-slim

# System libraries RDKit's drawing module links against (expat, X11 render);
# the slim base image has none of them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sar_app_cdd.py cdd_data.py ./
COPY assets ./assets

# Run unprivileged; /data is the only place the app writes.
RUN useradd --create-home --uid 10001 sar \
 && mkdir /data && chown sar:sar /data
USER sar
ENV SAR_DATA_DIR=/data \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]

EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/', timeout=4)"

# ONE worker: the data lives in this process's memory and the CDD download
# runs on a thread inside it. Threads serve concurrent users.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", \
     "--bind", "0.0.0.0:8050", "--access-logfile", "-", "sar_app_cdd:server"]
