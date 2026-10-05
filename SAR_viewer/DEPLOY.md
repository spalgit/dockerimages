# PROTAC SAR matrix — deployment brief

A small internal web app (Python / Dash) that shows a structure–activity matrix
of PROTAC compounds, built from assay data held in CDD Vault. This brief covers
how to run it and the security requirements the deployment must meet.

**The data is confidential. It must not be reachable by anyone outside the
company — see section 4. Those requirements are part of the acceptance test.**

---

## 1. What is in this package

| File | Purpose |
|---|---|
| `sar_app_cdd.py` | The web app (Dash). WSGI entry point: `sar_app_cdd:server` |
| `cdd_data.py` | Downloads the assay data from the CDD Vault API and keeps a local copy |
| `assets/` | CSS and browser scripts, served by the app |
| `requirements.txt` | Pinned Python dependencies (Python 3.12) |
| `Dockerfile` | Container image, ready to run (recommended way to deploy) |
| `.env.example` | The settings the app reads, with no values |

The package holds **code only — no data and no credentials**. The app fetches
its data from CDD Vault itself.

## 2. How it runs

- **Container:** `docker build -t sar-matrix .` The image runs gunicorn on
  port **8050** as an unprivileged user.
- **Exactly one worker process.** The data is held in that process's memory
  and the CDD download runs on a background thread inside it. Several workers
  would each hold and download their own copy. Concurrency comes from threads
  (`--threads 8`), which is plenty for an internal team. Do not scale beyond
  **one task / instance**.
- **Settings (environment variables):**

  | Variable | Value | Secret? |
  |---|---|---|
  | `cddVaultId` | CDD vault number | no |
  | `cddAPIToken` | CDD API token | **yes**: inject from AWS Secrets Manager |
  | `SAR_DATA_DIR` | `/data` (default in the image) | no |

- **Persistent storage:** the app writes its copy of the data
  (`cdd_sar_data.csv` and `cdd_sar_data.meta.json`, about 15 MB) to `/data`.
  Mount persistent storage there (e.g. EFS for ECS/Fargate), encrypted at rest.
  Without it the app still works, but after every restart it downloads again
  (about 1–2 minutes) before showing data.
- **First start:** with an empty `/data`, the first page visit starts the
  download in the background and the page fills in when it completes. Users can
  request fresh data at any time with the **Refresh from CDD** button.
- **Resources:** about 300 MB RAM at rest, more briefly during a download.
  0.5–1 vCPU and 2 GB RAM is ample.
- **Network:** inbound HTTP on 8050 from the load balancer only. Outbound
  HTTPS (443) to `app.collaborativedrug.com` only.
- **Health check:** `GET /` returns 200 (also defined in the Dockerfile).
  Allow a 30 s start-up grace period.
- **Logs:** gunicorn access and app logs go to stdout/stderr. Send them to
  CloudWatch.

Suggested AWS shape (adapt as you see fit, but the requirements in section 4
hold): **ECS on Fargate** (one task) in private subnets → **Application Load
Balancer** with HTTPS (ACM certificate) and OIDC sign-in → EFS for `/data` →
token from **Secrets Manager** → outbound access through a NAT gateway. Region:
**us-east-1** unless the client specifies otherwise.

## 3. Credentials

- The CDD API token is provided by the client **directly into AWS Secrets
  Manager** in the client's account, or through a one-time secure share. It is
  never sent by email or chat, and never stored in the image, the repository,
  a task-definition plain-text variable or logs.
- It is a dedicated token for this service. It can be revoked and reissued
  without affecting anyone else.

## 4. Security requirements (mandatory)

1. **AWS account:** everything is deployed in the **client's AWS account**,
   accessed through an IAM role the client grants and can revoke. No copy of
   the app, the data or the token is kept in Boulder Bio's own accounts or
   machines after deployment.
2. **Company sign-in only:** every request must be authenticated against the
   client's identity provider (Google Workspace or Microsoft 365 — to be
   confirmed with the client), for example ALB OIDC authentication or AWS
   Verified Access. Access is limited to a group the client controls. Accounts
   outside the company domain (including personal Gmail or Outlook) must be
   refused.
3. **HTTPS only:** HTTP redirects to HTTPS, using an ACM certificate on a
   company hostname.
4. **No direct path to the app:** the task or instance has no public IP and
   sits in a private subnet. Its security group accepts port 8050 only from the
   load balancer's security group. Admin access, if any, uses SSM Session
   Manager, not open SSH.
5. **Least privilege:** the task role can read only this one secret and write
   logs. No wildcard IAM permissions.
6. **Encryption at rest:** EFS, logs and the secret are encrypted (AWS defaults
   or KMS).
7. **Audit:** load-balancer access logs and CloudTrail are enabled, so we can
   see who signed in and when.
8. **Data handling:** do not copy `/data` or exports out of the account, for
   example for debugging, without the client's written agreement. This is
   covered by the NDA / services agreement.

## 5. Acceptance test

- [ ] A company account in the allowed group signs in and sees the matrix.
- [ ] A company account **not** in the group is refused.
- [ ] A personal Gmail or Outlook account is refused.
- [ ] Requests to the task or instance address that bypass the load balancer
      time out or are refused.
- [ ] `http://` redirects to `https://`.
- [ ] **Refresh from CDD** completes, and the "CDD data downloaded …" note
      shows the new time.
- [ ] Restarting the task keeps the data: no re-download needed, and the note
      shows the earlier download time.
- [ ] Find compound: entering `A645` (row) and `A095` (column) and pressing
      Enter jumps to that cell and highlights it.
- [ ] Hovering a matrix cell shows its dose-response curve.
- [ ] The CDD token appears nowhere in the image, logs or task-definition
      plain-text variables.

## 6. Handover back to the client

Please provide:

- Infrastructure as code (Terraform, CloudFormation or CDK) for everything
  created, in the client's repository.
- A one-page runbook covering: how to deploy a new version of the code, how to
  add or remove users, how to rotate the CDD token, and where the logs are.
- Confirmation that Boulder Bio's access role has been removed, or the agreed
  date it will be.

## Notes for developers

- `openpyxl` is imported before RDKit's drawing module on purpose. Reversing
  the order makes RDKit and openpyxl segfault. Keep the import order in
  `sar_app_cdd.py`.
- Local run without Docker: `pip install -r requirements.txt`, set the
  variables, then `python sar_app_cdd.py` (Dash development server on
  127.0.0.1:8050) or the gunicorn command from the Dockerfile.
- `python sar_app_cdd.py --offline` serves an existing local copy without
  contacting CDD.
