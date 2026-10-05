# Running the SAR matrix app on a GCP VM

How to build, run, restart and open the SAR matrix container on a Google Cloud
VM. The app stays private: it listens only on the VM's `127.0.0.1`, and you
reach it from your own computer through an SSH tunnel.

> Add `sudo` in front of the `docker` commands if you get
> `permission denied ... docker.sock`, or add yourself to the docker group once:
> `sudo usermod -aG docker $USER`, then log out and back in.

---

## 1. One-time setup

### 1.1 Install Docker on the VM (skip if `docker --version` works)
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
```

### 1.2 Get the code onto the VM
On the VM:
```bash
git clone git@github.com:spalgit/dockerimages.git
cd dockerimages/SAR_viewer
```

### 1.3 Create the credentials file
```bash
cp .env.example sar.env
nano sar.env          # fill in cddVaultId=... and cddAPIToken=...
chmod 600 sar.env
```
Never commit `sar.env` or copy it into the image.

---

## 2. Build and start the container

Run from the folder that contains the `Dockerfile`:
```bash
docker build -t sar-matrix .

docker run -d --name sar --restart unless-stopped \
    --env-file sar.env -v sar-data:/data \
    -p 127.0.0.1:8050:8050 sar-matrix
```

| Option | Why |
|---|---|
| `--name sar` | Lets you refer to the container as `sar` |
| `--restart unless-stopped` | Starts again automatically after a crash or VM reboot |
| `--env-file sar.env` | Passes in the CDD vault id and API token |
| `-v sar-data:/data` | Keeps the downloaded CDD data when the container is replaced |
| `-p 127.0.0.1:8050:8050` | Only the VM itself can reach port 8050. Don't drop the `127.0.0.1`, because the app has no login. |

---

## 3. Check that it is running

```bash
docker ps -a
```
Expected output:
```
IMAGE        STATUS                    PORTS                      NAMES
sar-matrix   Up 2 minutes (healthy)    127.0.0.1:8050->8050/tcp   sar
```
Look at the PORTS column. It must start with `127.0.0.1`. If it shows
`0.0.0.0:8050`, anyone who can reach the VM can open the app.

Follow the logs (press Ctrl+C to stop following; the app keeps running):
```bash
docker logs -f sar
```

---

## 4. Open the app from your computer

Run this on your **local machine**, not on the VM, and keep the window open:
```bash
gcloud compute ssh vm-gpu --zone YOUR_ZONE -- -L 8050:127.0.0.1:8050
```
Then open **http://localhost:8050** in your browser.

If you don't use `gcloud` locally, plain SSH works too:
```bash
ssh -L 8050:127.0.0.1:8050 spal@VM_EXTERNAL_IP
```

---

## 5. Day-to-day operations

| Task | Command |
|---|---|
| Restart (same code) | `docker restart sar` |
| Stop | `docker stop sar` |
| Start a stopped container | `docker start sar` |
| Status / health | `docker ps -a` |
| Logs | `docker logs -f sar` (or `docker logs --tail 100 sar`) |

After `docker restart`, wait about 30 seconds for the STATUS to show `(healthy)`.

### Deploying new code
`docker restart` keeps running the old image. When the code changes, pull the
new code, then rebuild and recreate the container:
```bash
cd ~/dockerimages/SAR_viewer
git pull
docker build -t sar-matrix .
docker rm -f sar
docker run -d --name sar --restart unless-stopped \
    --env-file sar.env -v sar-data:/data \
    -p 127.0.0.1:8050:8050 sar-matrix
```
The downloaded data in the `sar-data` volume is kept.

### Clearing the cached CDD data
Use this only if you want the app to download the data from CDD again:
```bash
docker rm -f sar
docker volume rm sar-data
# then run the `docker run ...` command from section 2 again
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `permission denied ... docker.sock` | Use `sudo docker ...`, or see the note at the top |
| `Conflict. The container name "/sar" is already in use` | `docker rm -f sar`, then run again |
| STATUS shows `(unhealthy)` or `Exited` | `docker logs --tail 100 sar`. Usually a wrong or missing `cddVaultId` / `cddAPIToken` in `sar.env` |
| Browser can't connect to `localhost:8050` | Check that the SSH tunnel window is still open and that `docker ps` shows the container `Up` |
| `bind: address already in use` on your laptop | Something local already uses port 8050. Tunnel to another port instead: `-L 8051:127.0.0.1:8050`, then open http://localhost:8051 |

## Security notes
- Keep the port bound to `127.0.0.1`. The app has **no authentication**.
- Docker publishes ports by writing its own iptables rules, so `ufw` does not
  block them. Don't rely on `ufw` to protect a port published on `0.0.0.0`.
- Don't add a GCP firewall rule that opens port 8050 to the internet. Use the
  SSH tunnel.
- `sar.env` holds the CDD API token: keep it at `chmod 600` and don't share it.
