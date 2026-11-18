# 4. AWS EC2 public server

Shell Buddy ran on a rented cloud server, so anyone with an API key could use it over HTTPS **without installing the model**. Only the small `howto.py` script runs on their laptop: it gets the suggestion from the server and runs the command locally after they approve it.

```
Friend's laptop ──HTTPS + API key──► Caddy (443) ──► gateway.py (8000) ──► Ollama (11434)
howto.py                             encrypts        checks key,           runs the model,
                                                     20 requests/min       hidden from the internet
```

| | |
|---|---|
| **Where the model runs** | A rented AWS server (Ubuntu 24.04, 2 CPUs, 4 GB RAM, `c7i-flex.large`) |
| **Cost** | ~$0.10/hour while running. Even when stopped, ~$5–6/month for the disk and the fixed IP. |
| **Speed** | ~1.7 s per request |
| **Status** | **Shut down** in November 2026: it used up free credits, and approach 5 does the job for free. Everything needed to rebuild it is here. |

**When to choose it:** when users must not have to download anything, or when you need one central place to control access (per-person keys you can revoke).

## Files

| File | Goes to (on the server) | Purpose |
|---|---|---|
| `gateway.py` | `/home/ubuntu/gateway/` | The "front desk": allows one action only (suggest a command), checks the API key, limits each key to 20 requests/minute, and hides FastAPI's docs pages |
| `shell-buddy.service` | `/etc/systemd/system/` | Keeps the gateway running, restarts it if it crashes, starts it on boot |
| `Caddyfile` | `/etc/caddy/Caddyfile` | HTTPS with a free, auto-renewed Let's Encrypt certificate; rejects requests over 10 KB; forwards to the gateway |

**Not in this repo, on purpose:** `/etc/shell-buddy.env`, the file holding the API keys (`SHELL_BUDDY_KEYS=key1,key2`). It's readable only by root, and systemd passes it to the gateway. **Never commit keys.**

## Why not open Ollama to the internet directly?

Ollama has **no password**. Anyone who finds it can use your server, download or delete models, and send prompts of any length. The gateway allows exactly one action, and only with a valid key. Ports 8000 and 11434 stay closed in the firewall.

## Rebuild it (summary)

The full click-by-click version, with explanations, is in [SHELL_BUDDY_GUIDE.md](../../SHELL_BUDDY_GUIDE.md#part-4-put-it-in-the-cloud), AWS Lessons A–F.

**A. Account safety first:** turn on MFA, and create a **$0.01 budget alert** so you get an email the moment anything costs money.

**B. Launch the server.** EC2 → Launch instance: Ubuntu 24.04, `c7i-flex.large`, 20 GB disk, your own SSH key, and **SSH from "My IP" only**. Allocate an **Elastic IP** (a fixed address) and attach it.

**C. Install Ollama and the model** (on the server):
```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/SushmithaJujare/shell-buddy-GGUF
ollama cp hf.co/SushmithaJujare/shell-buddy-GGUF shell-buddy     # the name gateway.py asks for
```

**D. Install the gateway** (from your laptop, then on the server):
```bash
scp deploy/4-aws-ec2/gateway.py deploy/4-aws-ec2/shell-buddy.service deploy/4-aws-ec2/Caddyfile ubuntu@<ELASTIC_IP>:~/
```
```bash
mkdir -p ~/gateway && mv ~/gateway.py ~/gateway/
sudo apt install -y python3-venv caddy
python3 -m venv ~/gateway/venv
~/gateway/venv/bin/pip install fastapi==0.162.0 uvicorn==0.54.0

# Create an API key, stored in a file only root can read
KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "SHELL_BUDDY_KEYS=$KEY" | sudo tee /etc/shell-buddy.env > /dev/null
sudo chmod 600 /etc/shell-buddy.env
echo $KEY            # save it in a password manager

sudo mv ~/shell-buddy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now shell-buddy
```

**E. HTTPS.** In the security group, open **HTTP (80) and HTTPS (443)** to anywhere, and keep SSH on My IP. Then:
```bash
sed -i "s/<ELASTIC_IP>/YOUR.ELASTIC.IP/" ~/Caddyfile
sudo mv ~/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```
[sslip.io](https://sslip.io) makes `<ELASTIC_IP>.sslip.io` point to your IP, so Caddy can get a certificate without you buying a domain.

**F. Test from your laptop:**
```bash
HOST=<ELASTIC_IP>.sslip.io
curl -s https://$HOST/health                                         # {"status":"ok"}
curl -s -X POST https://$HOST/suggest -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" -d '{"request":"show free memory"}'      # {"command":"free -m"}
curl -s -m 5 http://<ELASTIC_IP>:11434; echo " exit=$?"              # must time out (exit=28)
```

## Using it with howto

Each user saves **their own** key and points `howto` at the server:

```bash
nano ~/.shell_buddy_key          # paste the key on one line
chmod 600 ~/.shell_buddy_key
SHELL_BUDDY_GATEWAY=https://<ELASTIC_IP>.sslip.io howto "show free memory"
```

**Add a key for a friend:** append `,newkey` to the `SHELL_BUDDY_KEYS` line in `/etc/shell-buddy.env`, then run `sudo systemctl restart shell-buddy`. **Revoke a key:** delete it from that line and restart. One key per person means you can cancel one without affecting anyone else.

**Private alternative (only you):** skip Caddy and the gateway, and open an SSH tunnel to the server's Ollama:
```bash
ssh -N -L 11435:127.0.0.1:11434 ubuntu@<ELASTIC_IP>
SHELL_BUDDY_URL=http://127.0.0.1:11435 SHELL_BUDDY_MODEL=shell-buddy howto "show free memory"
```

## Shutting it down so you stop paying

**Stopping** the server isn't enough: the disk and the Elastic IP are still charged. To end the costs:

1. **Terminate** the instance (this deletes the server and its disk)
2. **Release** the Elastic IP. This is the most-forgotten step.
3. Check that **Volumes** is empty
4. Check the bill a day or two later
