# Giving colleagues secure access to the SAR app with Tailscale

The SAR app runs in Docker on the VM `vm-sar-viewer` and listens only on the
VM's `127.0.0.1:8050` (see [RUN_ON_GCP_VM.md](RUN_ON_GCP_VM.md)). Tailscale
puts the app on ChiaraBio's private network: only approved people can open
it. The VM opens no port to the internet.

```
Colleague's laptop ──(encrypted Tailscale link)──> vm-sar-viewer ──> 127.0.0.1:8050 (Docker: sar)
```

**ChiaraBio's Tailscale setup**

| | |
|---|---|
| Tailnet | `chiarabio.com` (names end in `tail7520f8.ts.net`) |
| Sign-in | **Microsoft 365** single sign-on with a ChiaraBio account |
| Tailnet admin | Kirsten (approves new users) |
| VM | `vm-sar-viewer`, Tailscale address `100.109.144.77` |
| App URL | `https://vm-sar-viewer.tail7520f8.ts.net` (once `serve` is set up, A4) |

- **Part A** is for whoever sets up the VM. It's done once.
- **Part B** is for each colleague. You can send them that section as it is.
- **Part C** covers testing, removing access and troubleshooting.

---

## Part A — Setup (once)

### A1. Get approved on the ChiaraBio tailnet
Joining needs two approvals: Microsoft 365 (done once for the company) and
a separate per-user approval on Tailscale's own website. See
[TAILSCALE_USER_APPROVAL.md](TAILSCALE_USER_APPROVAL.md).

To change access rules or DNS settings yourself, ask the tailnet admin to
give you the **Network admin** role (Users → **⋯** → Edit role).

### A2. Install Tailscale on the VM
SSH into the VM, then:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
`tailscale up` prints a login link. Open it in your browser and sign in with
**Sign in with Microsoft** using your ChiaraBio account. `Success.` means
the sign-in worked. The VM is only usable once your user is approved (A1).

Check:
```bash
tailscale status --json | grep MagicDNSSuffix     # expect tail7520f8.ts.net
```
Tailscale starts automatically when the VM reboots.

### A3. Turn on HTTPS names (admin or network admin)
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
The status command shows the app's private address:
```
https://vm-sar-viewer.tail7520f8.ts.net (tailnet only)
|-- / proxy http://127.0.0.1:8050
```
Colleagues open this URL. The setting survives reboots.

Check it from the VM:
```bash
curl -sI https://vm-sar-viewer.tail7520f8.ts.net     # expect: HTTP/2 200
```

> **Never run `tailscale funnel`.** It publishes the app to the whole
> internet. `serve` keeps it private.

### A5. Give colleagues access
How you do this depends on whether the person has a ChiaraBio account.

**ChiaraBio colleagues (`@chiarabio.com`)** join the company tailnet directly:
1. They follow **Part B**. When they first sign in, they see
   "User approval required".
2. The tailnet admin approves them: Console → **Users** → their row →
   **⋯ → Approve** (details in
   [TAILSCALE_USER_APPROVAL.md](TAILSCALE_USER_APPROVAL.md)).
3. If the tailnet's **Access controls** restrict who can reach what, they
   also need a rule that allows the VM (A6).

**People outside ChiaraBio** (e.g. a client) can't join the company tailnet.
Share the VM with them instead:
Console → **Machines** → **⋯** on the `vm-sar-viewer` row → **Share…** →
enter their email, or use **Copy share link** and send it yourself. Sharing
gives access to this one machine only. Don't invite outsiders under
**Users → Invite users**, which makes them members of the whole network.

### A6. Access rule for the SAR viewer (only if needed)
If approved ChiaraBio users still can't reach the VM (`tailscale ping
vm-sar-viewer` prints `no matching peer`), the policy in Console →
**Access controls** is blocking them. An admin or network admin can add a
group and a rule for the viewer. **Read the existing policy first**, and
add to it rather than replacing it:
```json
"groups": {
  "group:sar-users": ["spal@chiarabio.com", "colleague@chiarabio.com"]
},
"hosts": {
  "vm-sar-viewer": "100.109.144.77"
},
"grants": [
  {"src": ["group:sar-users"], "dst": ["vm-sar-viewer"], "ip": ["443"]}
]
```
Port 443 is enough for the app through `serve`. To give yourself SSH or
other access to the VM as well, add a separate rule for your own address.
To add a new colleague later, add their address to `group:sar-users`.

### A7. Billing
Each ChiaraBio user on the tailnet counts toward the company's Tailscale
plan, which the tailnet admin manages (**Settings → Billing**). Check with
them before adding many users.

---

## Part B — Instructions for ChiaraBio colleagues (send this part)

> **Accessing the SAR Matrix app**
>
> You will need about 5 minutes and a Windows laptop. No technical
> knowledge is needed.
>
> **1. Install Tailscale**
> - Download it from https://tailscale.com/download/windows and run the
>   installer. Click **Yes** if Windows asks for permission.
> - Tailscale doesn't open a window. A small icon (a grid of dots) appears
>   near the clock, bottom-right. Click **^** if you don't see it.
>
> **2. Sign in with your ChiaraBio account**
> Click the icon → **Log in**. In the browser, choose **Sign in with
> Microsoft** and use your **@chiarabio.com** account.
>
> If you see **"User approval required"**, that's expected the first time.
> Leave the page open and let Sandeep or Kirsten know. Once you're
> approved, you're logged in automatically.
>
> **3. Open the app**
> Open this address in Chrome or Edge and bookmark it:
>
> &nbsp;&nbsp;&nbsp;&nbsp;**https://vm-sar-viewer.tail7520f8.ts.net**
>
> **Every time you use the app:** make sure Tailscale is connected (click the
> tray icon; it should say **Connected**), then open your bookmark.
>
> **If the page doesn't load:** click the Tailscale icon and check that it says
> Connected and that you are signed in with your ChiaraBio account. Then
> refresh the page. If it still doesn't load, contact Sandeep.
>
> Please don't share the link or your login. The data is confidential.

For people outside ChiaraBio, change step 2 to: "Open the share link from
the email and click **Accept**, then sign in to Tailscale with the same
email address the invitation was sent to."

---

## Part C — Testing, removing access, troubleshooting

### C1. Test it from your laptop
1. On the laptop, Tailscale shows **Connected** with your ChiaraBio account.
2. In PowerShell:
   ```powershell
   & "C:\Program Files\Tailscale\tailscale.exe" ping vm-sar-viewer    # expect: pong
   ```
3. Open `https://vm-sar-viewer.tail7520f8.ts.net`. The SAR app should appear.
4. Click the Tailscale icon → **Disconnect**, then reload the page. It must
   **fail** to load, which shows that the app is reachable only through
   Tailscale. Reconnect afterwards.

### C2. Remove someone's access
- **ChiaraBio colleague:** remove them from `group:sar-users` (A6), or have
  the tailnet admin suspend or remove them under **Users**. People who leave
  ChiaraBio also lose access when their Microsoft account is disabled.
- **External person:** Console → **Machines** → **⋯** on `vm-sar-viewer` →
  **Share…** → remove the person.

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
| Browser shows **User approval required** | The user hasn't been approved in Tailscale yet. Approval in Microsoft 365 is not enough; see [TAILSCALE_USER_APPROVAL.md](TAILSCALE_USER_APPROVAL.md) |
| `tailscale up` says `Success.` but `tailscale status` lists only this machine | Same cause: the user is still waiting for Tailscale approval |
| `tailscale ping vm-sar-viewer` prints `no matching peer` after approval | Access controls block it (A6), or the two devices are in different tailnets (compare `MagicDNSSuffix`: both must be `tail7520f8.ts.net`) |
| Clicking the Tailscale app on Windows does nothing | It runs in the system tray: click **^** by the clock. If there's no icon, run `Start-Service Tailscale` in an admin PowerShell |
| Colleague's page never loads | Tailscale not connected on their laptop, or signed in with a personal account instead of their ChiaraBio one |
| `vm-sar-viewer` shows **Offline** in the console | On the VM: `sudo systemctl restart tailscaled`, then `tailscale status` |
| Page shows **502 Bad Gateway** | Tailscale works but the app is down: `docker ps -a`, `docker logs --tail 100 sar` |
| Certificate warning in the browser | HTTPS Certificates not enabled (A3). Wait a minute after enabling and retry |

---

## Security summary
- The app has **no login of its own**. Tailscale (approval, access rules and
  sharing) is what controls access, so grant access only to people approved
  to see the data.
- Sign-in goes through ChiaraBio's Microsoft 365, so its two-factor
  sign-in and account disabling apply to Tailscale too.
- Docker keeps the app on `127.0.0.1:8050`. Don't change the `-p` mapping and
  don't open port 8050 in the VM's cloud firewall.
- Use `serve`, never `funnel`.
- Traffic between the colleague and the VM is encrypted end to end.
  Tailscale's servers only coordinate the connection.
- Share the VM with outsiders only through **Share**, never by inviting them
  as tailnet users.
- Review `group:sar-users` and the share list from time to time, and remove
  people who no longer need access.
- Before sharing client CDD data with anyone, confirm the client approves,
  including access from other countries.
