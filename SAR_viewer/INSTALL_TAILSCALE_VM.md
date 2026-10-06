# Installing Tailscale on the VM and giving colleagues access

This guide puts the SAR Viewer on ChiaraBio's private Tailscale network
(tailnet `chiarabio.com`, names ending in `tail7520f8.ts.net`) so
colleagues can open it from their own laptops. Nothing is opened to the
public internet.

Before you start, the app must already be running on this VM
(see [RUN_ON_GCP_VM.md](RUN_ON_GCP_VM.md)):
```bash
docker ps        # container "sar" is Up (healthy), PORTS = 127.0.0.1:8050->8050/tcp
```

---

## 1. Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled
```

## 2. Connect the VM to the ChiaraBio tailnet

```bash
sudo tailscale up
```
- The command prints a login link (`https://login.tailscale.com/a/...`).
  Open it in your laptop's browser, choose **Sign in with Microsoft** and
  use your ChiaraBio account (`spal@chiarabio.com`).
- The machine name is the VM's hostname, `vm-sar-viewer`. It becomes part
  of the app URL. To use a different name, add `--hostname NAME`.
- If the browser shows **User approval required**, the tailnet admin
  (Kirsten) must approve you on Tailscale's website. Approval in Microsoft
  365 is not enough. See [TAILSCALE_USER_APPROVAL.md](TAILSCALE_USER_APPROVAL.md).
  The VM connects by itself once you're approved.

Check:
```bash
tailscale status                                   # this VM: 100.109.144.77  vm-sar-viewer
tailscale status --json | grep MagicDNSSuffix      # tail7520f8.ts.net
```
`sudo tailscale up` printing `Success.` only means the sign-in worked. If
`tailscale status` lists only this VM and none of your other devices, the
user approval is probably still pending.

The machine also appears in the console: https://login.tailscale.com/admin/machines

## 3. Turn on HTTPS (once for the tailnet)

Needs an Admin or Network admin. Console → **DNS**:
- **MagicDNS**: enabled
- **HTTPS Certificates**: **Enable HTTPS**

## 4. Put the app on the private network

```bash
sudo tailscale serve --bg 8050
sudo tailscale serve status
```
Expected output:
```
https://vm-sar-viewer.tail7520f8.ts.net (tailnet only)
|-- / proxy http://127.0.0.1:8050
```
That address is **the app URL** to give colleagues. The setting survives
reboots.

Test it from the VM (the first request may take a few seconds while the
certificate is issued):
```bash
curl -sI https://vm-sar-viewer.tail7520f8.ts.net      # expect: HTTP/2 200
```

> **Never run `tailscale funnel`.** It makes the app public on the internet.
> `serve` keeps it private.

---

## 5. Give colleagues access

### ChiaraBio colleagues
People with an `@chiarabio.com` account join the company tailnet themselves:
1. Send them **Part B** of [TAILSCALE_ACCESS.md](TAILSCALE_ACCESS.md)
   (install Tailscale for Windows → sign in with Microsoft → open the URL).
2. The first time they sign in, they see **User approval required**. The
   tailnet admin approves them under **Users** (see
   [TAILSCALE_USER_APPROVAL.md](TAILSCALE_USER_APPROVAL.md)).
3. If they're approved but still can't reach the app, add them to the
   access rule in Part A6 of [TAILSCALE_ACCESS.md](TAILSCALE_ACCESS.md).

### People outside ChiaraBio (sharing)
Outsiders can't join the company tailnet. Use **sharing**, which gives
access to this one VM only. Don't use *Users → Invite users*, which would
make them members of the whole network.

**Option A: from the admin console (simplest)**
1. Open https://login.tailscale.com/admin/machines
2. Click **⋯** on the `vm-sar-viewer` row → **Share…**
3. Enter their email and send. Or click **Copy share link** and email the
   link yourself.

**Option B: from this VM's terminal (Tailscale API)**

*One-time step: create an API access token.*
Console → **Settings → Keys** → **Generate access token** → copy it
(it starts with `tskey-api-`). This needs a role that is allowed to create
keys (ask the tailnet admin if the option is missing). The token carries
your permissions on the company tailnet, so never save it in this folder or
in git. Give it a short expiry (for example 7 days).

*Send an invite:*
```bash
read -s -p "Tailscale API token: " TS_API_KEY; echo
DEVICE_ID=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["ID"])')

curl -s -u "$TS_API_KEY:" \
  -H "Content-Type: application/json" \
  -X POST "https://api.tailscale.com/api/v2/device/$DEVICE_ID/device-invites" \
  -d '[{"email": "person@example.com"}]'
```
Tailscale emails them an invitation. The response contains an `inviteUrl`
you can also send them directly.

To invite several people, list them all:
`-d '[{"email": "a@example.com"}, {"email": "b@example.com"}]'`

*List invites for this VM:*
```bash
curl -s -u "$TS_API_KEY:" \
  "https://api.tailscale.com/api/v2/device/$DEVICE_ID/device-invites"
```

*Cancel a pending invite* (use the `id` from the list above):
```bash
curl -s -u "$TS_API_KEY:" -X DELETE \
  "https://api.tailscale.com/api/v2/device-invites/INVITE_ID"
```

When you've finished, clear the token from the shell with `unset TS_API_KEY`,
and revoke it in **Settings → Keys** if you won't need it again.

> If an API call returns `404` or `403`, check that the token hasn't expired
> and that your role allows sharing. If it still fails, use Option A or ask
> the tailnet admin.

---

## 6. Remove someone's access

- **ChiaraBio colleague:** remove them from `group:sar-users` in Access
  controls, or have the tailnet admin suspend or remove them under **Users**.
- **External person:** Console → **Machines** → **⋯** on `vm-sar-viewer` →
  **Share…** → remove them. Unaccepted invites can also be cancelled from
  the VM with the API `DELETE` call above.

Access stops immediately.

**Emergency off-switch** (takes the app off Tailscale for everyone):
```bash
sudo tailscale serve reset
```
Turn it back on with `sudo tailscale serve --bg 8050`.

---

## 7. Useful commands

| Task | Command |
|---|---|
| Is the VM connected? | `tailscale status` |
| Which tailnet is it in? | `tailscale status --json \| grep MagicDNSSuffix` |
| Can it reach another device? | `tailscale ping DEVICE-NAME` |
| What is being served? | `sudo tailscale serve status` |
| VM's Tailscale address | `tailscale ip -4` |
| Disconnect the VM from Tailscale | `sudo tailscale down` |
| Reconnect | `sudo tailscale up` |
| Restart the Tailscale service | `sudo systemctl restart tailscaled` |
| Remove the VM from the tailnet entirely | `sudo tailscale logout` |

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| **User approval required** in the browser | The tailnet admin must approve the user on login.tailscale.com → Users. See [TAILSCALE_USER_APPROVAL.md](TAILSCALE_USER_APPROVAL.md) |
| `tailscale status` lists only this machine; `tailscale ping` prints `no matching peer` | User approval still pending, the access rules block the devices, or they're in different tailnets (check `MagicDNSSuffix` on both) |
| Colleague sees nothing / page never loads | Their Tailscale app isn't connected, or they signed in with a personal account instead of their ChiaraBio one |
| **502 Bad Gateway** | Tailscale works but the app is down: `docker ps -a`, `docker logs --tail 100 sar` |
| Certificate warning | HTTPS Certificates not enabled (step 3). Wait a minute and retry |
| Machine shows **Offline** in console | `sudo systemctl restart tailscaled`, then `tailscale status` |
| Machine disconnected after ~180 days | Key expiry. Run `sudo tailscale up` again, or ask an admin to choose **⋯ → Disable key expiry** for this server in the console |

## Security checklist
- [ ] Docker port is `127.0.0.1:8050` (check `docker ps`); no cloud firewall rule opens 8050
- [ ] Only `serve` is used, never `funnel`
- [ ] ChiaraBio colleagues are approved users; outsiders are added by **Share**, never invited as users
- [ ] API token not saved anywhere; revoked when no longer needed
- [ ] Access group and share list reviewed regularly; leavers removed
- [ ] Client has approved who may see the CDD data
