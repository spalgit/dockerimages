# SAR Viewer — PROTAC SAR matrix

A Dash web app that downloads HiBiT and Jess assay data from a CDD Vault and
shows it as an interactive PROTAC SAR matrix. It runs as a single Docker
container.

This repository holds **code only**: no assay data and no credentials. The app
downloads its own copy of the CDD data into `/data` (a Docker volume) the
first time it runs.

## Quick start

```bash
cp .env.example sar.env      # fill in cddVaultId and cddAPIToken
chmod 600 sar.env

docker build -t sar-matrix .
docker run -d --name sar --restart unless-stopped \
    --env-file sar.env -v sar-data:/data \
    -p 127.0.0.1:8050:8050 sar-matrix
```
Then open http://127.0.0.1:8050 on the same machine, or reach it through an
SSH tunnel or Tailscale (see below).

## Files

| File | Purpose |
|---|---|
| `sar_app_cdd.py` | Dash app (gunicorn entry point `sar_app_cdd:server`) |
| `cdd_data.py` | CDD Vault download, caching and data preparation |
| `assets/` | CSS and browser scripts served by the app |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | Container image |
| `.env.example` | Template for the credentials file (never commit the real one) |
| `RUN_ON_GCP_VM.md` | Build, run, restart and update on a GCP VM |
| `TAILSCALE_ACCESS.md` | Giving colleagues private access with Tailscale |
| `DEPLOY.md` | Production (AWS) deployment and security requirements |

## Security
- The app has **no built-in login**. Keep the port bound to `127.0.0.1` and
  give access only through an SSH tunnel, Tailscale `serve`, or an
  authenticating load balancer.
- `sar.env` / `.env`, and any `*.csv` / `*.xlsx` data files, are excluded by
  `.gitignore`. Never commit them.

## Local run without Docker
```bash
pip install -r requirements.txt
export cddVaultId=... cddAPIToken=...
python sar_app_cdd.py            # serves on 127.0.0.1:8050
```
Note: `openpyxl` is imported before RDKit's drawing module on purpose.
Reversing the order makes them segfault.
