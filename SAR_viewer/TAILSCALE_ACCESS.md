# Giving colleagues secure access to the SAR app with Tailscale

The SAR app runs in Docker on the GCP VM `vm-gpu` and listens only on the VM's
`127.0.0.1:8050` (see [RUN_ON_GCP_VM.md](RUN_ON_GCP_VM.md)). Tailscale puts the
app on a private network: only people you have shared the VM with can open it.
The VM opens no port to the internet.

```
Colleague's laptop ──(encrypted Tailscale link)──> vm-gpu ──> 127.0.0.1:8050 (Docker: sar)
```

- **Part A** is for you, the admin. You do it once.
- **Part B** is for each colleague. You can send them that section as it is.
- **Part C** covers testing, removing access and troubleshooting.

---

## Part A — Admin setup (once)

### A1. Create the Tailscale account
1. Go to https://login.tailscale.com and sign in with your work account
   (`sandeep@cheminfosolutions.co.uk`).
2. Answer the welcome questions however you like. They don't change any
   settings.

### A2. Install Tailscale on the VM
SSH into the VM, then:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
`tailscale up` prints a login link. Open it in your browser and approve the
device. `vm-gpu` now appears under **Machines** with status **Connected**.
Tailscale starts automatically when the VM reboots.

### A3. Turn on HTTPS names
Console → **DNS**:
- **MagicDNS**: enabled (usually on by default)
- **HTTPS Certificates**: click **Enable HTTPS**

### A4. Put the app on the private network
Make sure the container is running (`docker ps` shows `sar` as `healthy`),
then on the VM:
```bash
sudo tailscale serve --bg 8050
sudo tailscale serve status
```
The status command shows the app's private address, for example:
```
https://vm-gpu.tailXXXX.ts.net (tailnet only)
|-- / proxy http://127.0.0.1:8050
```
Write down this URL, because colleagues will open it. The setting survives
reboots.

Check it from the VM:
```bash
curl -sI https://vm-gpu.tailXXXX.ts.net     # expect: HTTP/2 200
```

> **Never run `tailscale funnel`.** It publishes the app to the whole
> internet. `serve` keeps it private.

### A5. Invite a colleague (share the VM)
Console → **Machines** → **⋯** on the `vm-gpu` row → **Share…**
- Enter the colleague's email address and send, or use **Copy share link**
  and send them the link yourself.
- Then send them **Part B** below together with the app URL from A4.

Always use **Share**, not **Users → Invite users**. Sharing gives access to
this one VM only. Inviting makes the colleague a member of your whole
network.

### A6. Check your plan before the trial ends
The console shows "Trial — 14 days left". Before it expires, check
**Settings → Billing** to see which plan the network will move to, so
colleagues don't lose access unexpectedly.

---

## Part B — Instructions for colleagues (send this part)

> **Accessing the SAR Matrix app**
>
> You will need about 5 minutes and a Windows laptop. No technical
> knowledge is needed.
>
> **1. Accept the invitation**
> Open the email from Tailscale (or the link Sandeep sent you) and click
> **Accept**. Sign in with the email address the invitation was sent to,
> using "Sign in with Google" or "Sign in with Microsoft".
>
> **2. Install Tailscale**
> - Download it from https://tailscale.com/download/windows and run the
>   installer. Click **Yes** if Windows asks for permission.
> - A small Tailscale icon appears near the clock (bottom-right). Click
>   **^** if you don't see it.
> - Click the icon → **Log in**, and sign in with the **same account** as in
>   step 1.
>
> **3. Open the app**
> Open this address in Chrome or Edge and bookmark it:
>
> &nbsp;&nbsp;&nbsp;&nbsp;**https://vm-gpu.tailXXXX.ts.net**
>
> **Every time you use the app:** make sure Tailscale is connected (click the
> tray icon; it should say **Connected**), then open your bookmark.
>
> **If the page doesn't load:** click the Tailscale icon and check that it says
> Connected and that you are signed in with the invited account. Then
> refresh the page. If it still doesn't load, contact Sandeep.
>
> Please don't share the link or your login. The data is confidential.

---

## Part C — Testing, removing access, troubleshooting

### C1. Test it yourself as a "colleague"
Use a second Google account to go through exactly what a colleague will:
1. Share `vm-gpu` to the second account (A5).
2. In a Chrome **Incognito** window, open the share link and accept it,
   signed in as the second account.
3. Install Tailscale on your laptop and log in as the **second account**.
4. Open the app URL. The SAR app should appear.
5. Click the Tailscale icon → **Log out**, then reload the URL. It must
   **fail** to load, which shows that only shared accounts can reach the app.
6. Revoke the test share (C2).

### C2. Remove someone's access
Console → **Machines** → **⋯** on `vm-gpu` → **Share…** → remove the person.
Access stops immediately.

### C3. Day-to-day
| Task | Command (on the VM) |
|---|---|
| Is Tailscale connected? | `tailscale status` |
| What is being served? | `sudo tailscale serve status` |
| Stop serving the app (emergency off-switch) | `sudo tailscale serve reset` |
| Serve it again | `sudo tailscale serve --bg 8050` |
| Restart the app | `docker restart sar` |

Restarting or rebuilding the Docker container doesn't affect Tailscale,
as long as the container still publishes `127.0.0.1:8050`.

### C4. Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Colleague's page never loads | Tailscale not connected on their laptop, or they're signed in with a different account than the one invited |
| `vm-gpu` shows **Offline** in the console | On the VM: `sudo systemctl restart tailscaled`, then `tailscale status` |
| Page shows **502 Bad Gateway** | Tailscale works but the app is down: `docker ps -a`, `docker logs --tail 100 sar` |
| Certificate warning in the browser | HTTPS Certificates not enabled (A3). Wait a minute after enabling and retry |
| Colleague can see other machines of yours | They were invited as a user instead of shared. Remove them under **Users** and share the machine instead (A5) |

---

## Security summary
- The app has **no login of its own**. Tailscale sharing is what controls
  access, so share only with people approved to see the data.
- Docker keeps the app on `127.0.0.1:8050`. Don't change the `-p` mapping and
  don't open port 8050 in the GCP firewall.
- Use `serve`, never `funnel`.
- Traffic between the colleague and the VM is encrypted end to end.
  Tailscale's servers only coordinate the connection.
- Ask colleagues to use two-factor sign-in on the account they use with
  Tailscale.
- Review the share list from time to time, and remove people who no longer
  need access.
- Before sharing client CDD data with anyone, confirm the client approves,
  including access from other countries.
