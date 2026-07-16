# Deploying lumi to Oracle Cloud (Always Free)

This replaces the Render deployment. It uses:
- An Oracle Cloud **Always Free** Ampere A1 VM (4 OCPUs / 24GB RAM available in
  the free tier — massively more than Render's free plan, and it never spins
  down or sleeps).
- Plain **systemd + uvicorn** (no Docker, since you just want free/simple).
- **Caddy** as a reverse proxy for free automatic HTTPS.
- **MongoDB Atlas** (free M0 cluster) for the database, as set up above.

---

## 1. Create the VM

1. Sign up / log into https://cloud.oracle.com (a credit card is required for
   verification, but Always Free resources are never billed).
2. **Compute -> Instances -> Create Instance**.
3. Image: **Ubuntu 24.04** (Canonical Ubuntu, "Always Free Eligible").
4. Shape: **VM.Standard.A1.Flex** (Ampere/ARM) — this is the Always Free
   shape. Set 2-4 OCPUs / 12-24GB RAM (still free, up to 4 OCPU/24GB total
   across all your A1 instances).
5. Networking: use the default VCN, and make sure "Assign a public IPv4
   address" is checked.
6. Add your SSH public key (or download the generated key pair).
7. Create the instance, note its **public IP**.

## 2. Open the firewall (both layers)

Oracle Cloud has two firewalls you must open — people forget the second one
and then can't figure out why the site is unreachable:

**a) Security List / Network Security Group** (Networking -> Virtual Cloud
Networks -> your VCN -> Security Lists -> Default Security List):
- Add ingress rules for TCP **80** and **443** from `0.0.0.0/0`.

**b) The VM's own OS firewall** (Ubuntu ships with iptables rules that block
everything but SSH by default on OCI images):
```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save   # or: sudo iptables-save > /etc/iptables/rules.v4
```

## 3. Install dependencies on the VM

```bash
ssh ubuntu@<your-vm-ip>

sudo apt update && sudo apt install -y python3-pip python3-venv git nginx-light

# Caddy (free, automatic HTTPS via Let's Encrypt)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

Uninstall nginx-light again if you installed it by mistake — you only need
one reverse proxy; these instructions use Caddy.

## 4. Get the code onto the VM

```bash
cd /opt
sudo mkdir lumi && sudo chown ubuntu:ubuntu lumi
git clone <your-repo-url> lumi   # or scp the project folder up
cd lumi

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your real `MONGODB_URI`,
`CEREBRAS_API_KEY`, `RESEND_API_KEY`, etc:

```bash
cp .env.example .env
nano .env
```

Warm the embedding model cache once so the very first real request isn't slow:

```bash
python3 scripts/warm_embedding_cache.py
```

If you're migrating existing data (see the main migration steps above),
run `scripts/migrate_to_mongo.py --all` once here or from your laptop
(anywhere that can reach both the old Postgres DB and the new Atlas
cluster).

## 5. systemd service

Create `/etc/systemd/system/lumi.service`:

```ini
[Unit]
Description=lumi FastAPI backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/lumi
Environment=PATH=/opt/lumi/.venv/bin
EnvironmentFile=/opt/lumi/.env
ExecStart=/opt/lumi/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lumi
sudo systemctl status lumi     # confirm it's running
journalctl -u lumi -f          # tail logs
```

Note `--workers 2`: each uvicorn worker is a separate process with its own
in-memory caches (embedding model, Chroma-replacement connections, mentor
retriever singleton). That's fine — each one warms up once on boot and then
stays warm, since this VM never sleeps like Render's free tier did.

## 6. Caddy reverse proxy + free HTTPS

If you have a domain, point an A record at the VM's public IP, then edit
`/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy 127.0.0.1:8000
}
```

No domain yet? Use the IP directly (HTTP only, no TLS):

```
:80 {
    reverse_proxy 127.0.0.1:8000
}
```

```bash
sudo systemctl restart caddy
```

Your API is now live at `https://your-domain.com` (or `http://<vm-ip>`),
staying up permanently — no more cold starts.

## 7. MongoDB Atlas network access

In Atlas -> **Network Access**, add the VM's public IP (or `0.0.0.0/0` if
you want to allow from anywhere, less secure but fine for a small personal
project). Without this, `pymongo.MongoClient` will hang for ~8s per request
and then fail with a server-selection timeout.

## 8. Redeploying after code changes

```bash
cd /opt/lumi
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart lumi
```

---

## What changed vs. the Render setup, and why it's faster

- **No more cold starts.** Render's free tier spins your service down after
  ~15 minutes idle; every wake-up re-downloaded the ~90MB ONNX embedding
  model and rebuilt the in-memory mentor index from scratch. The OCI VM
  never sleeps, so both stay warm indefinitely after the first request.
- **Contest embeddings now live in MongoDB Atlas** (`contest_chunks`
  collection with an Atlas Vector Search index), not a local ChromaDB
  folder. They're computed once at ingest time and never recomputed at
  query time — only the (tiny, ~1-5ms) query-text embedding happens per
  request, which is unavoidable for any RAG system.
- **Contest PDFs live in MongoDB GridFS**, with a local disk cache
  (`data/pdf_cache/`) so repeated requests for the same PDF are a plain
  file read, not a database round-trip.
- **Bookings/mentors/mentees/timeslots live in MongoDB** instead of Neon
  Postgres — see `api/db.py`, `api/services.py`, `api/admin.py`.

## Known follow-up (not migrated in this pass)

`models/clean_training_data.py` and `scripts/import_and_train.py` are
offline maintenance tools (not on the live request path) that still assume
a SQL database with `information_schema` introspection and dynamic column
queries against a separate `historical_pairings` table. They weren't
rewritten for MongoDB in this pass — if you actively use them to import new
training CSVs, they'll need a similar but separate rewrite. Everything on
the live request path (matching, booking, admin panel, contest chat/search,
problem-set PDFs) is fully migrated.
