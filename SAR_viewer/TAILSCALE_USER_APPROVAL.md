# Approving a new user in Tailscale (ChiaraBio)

ChiaraBio's Tailscale network (tailnet `chiarabio.com`, names ending in
`tail7520f8.ts.net`) requires **two separate approvals** before someone can
use it:

| | Where | What it controls |
|---|---|---|
| 1 | **Microsoft 365** (Entra ID → Enterprise Applications → Tailscale) | Whether ChiaraBio accounts can **sign in** to Tailscale with Microsoft. Done once for the whole company. |
| 2 | **Tailscale admin console** (https://login.tailscale.com) | Whether each signed-in person may **join** the network. Done once per new user. |

Approval 1 does not cover approval 2. A user who has only approval 1 can sign
in, and `sudo tailscale up` prints `Success.`, but their devices stay on hold:
each one only sees itself in `tailscale status`, and `tailscale ping` to any
other device prints `no matching peer`.

---

## For the admin: approve the user (about 1 minute)

1. Go to **https://login.tailscale.com** (not the Microsoft 365 portal).
2. Sign in with your **ChiaraBio account**. Tailscale sends you through the
   usual Microsoft login.
3. Click **Users** at the top of the page.
4. Find the person's address (for example `spal@chiarabio.com`). It has a
   **Needs approval** label.
5. Click **⋯** (three dots) at the end of that row → **Approve**.

The user's devices connect automatically. Nobody has to sign in again.

### Optional: let the user manage the setup
On the same row, click **⋯ → Edit role** → **Network admin** → **Save**.
A network admin can change access rules and DNS settings (for example,
giving colleagues access to the SAR viewer) but cannot manage billing or
users.

### If a device also needs approval
If the tailnet also requires device approval, open **Machines**, find the
device (for example `vm-sar-viewer`), and click **⋯ → Approve**.

---

## For the user: what you see before approval

When you sign in, the browser shows:

> **User approval required**
> Admins of this tailnet must approve you before you can join.

Leave that page open. Once the admin approves you, it logs you in by itself.

## Checking that it worked

On the VM:
```bash
tailscale status        # should now list your laptop as well as vm-sar-viewer
```

On a Windows laptop (PowerShell):
```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
& "C:\Program Files\Tailscale\tailscale.exe" ping vm-sar-viewer
```
`ping` should reply `pong`. If both machines are approved but still can't see
each other, the tailnet's **Access controls** are blocking them, and an admin
or network admin must add a rule.

To check that both machines are in the same tailnet:
```bash
tailscale status --json | grep MagicDNSSuffix                                          # VM
& "C:\Program Files\Tailscale\tailscale.exe" status --json | Select-String MagicDNSSuffix  # Windows
```
Both should show `tail7520f8.ts.net`.

After that, continue with [INSTALL_TAILSCALE_VM.md](INSTALL_TAILSCALE_VM.md)
(HTTPS and `tailscale serve`).
