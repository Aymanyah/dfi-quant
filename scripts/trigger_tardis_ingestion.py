"""Déclenche l'ingestion Tardis sur le JupyterHub et télécharge les parquet résultants.

Usage:
    python scripts/trigger_tardis_ingestion.py \
        --from 2024-03-01 --to 2024-03-07 \
        --symbols BTCUSDT

Le script:
1. Crée un kernel Python sur le JupyterHub
2. Exécute agents.tardis_ingestion.runner sur le serveur
3. Surveille la progression en temps réel
4. Télécharge les fichiers parquet résultants localement
5. Supprime le kernel
"""
from __future__ import annotations
import argparse, base64, json, pathlib, ssl, time, uuid
import urllib.request
import websocket

HUB      = "https://103.16.231.53.nip.io"
TOKEN    = "2886fd80ad4e42188efe9ac3cf013484"
USER     = "dfi"
BASE_API = f"{HUB}/user/{USER}/api"
RAW_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Content-Type":  "application/json",
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(method: str, path: str, body: dict | None = None) -> dict:
    url  = BASE_API + path
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _create_kernel() -> str:
    resp = _req("POST", "/kernels", {"name": "python3"})
    kid  = resp["id"]
    print(f"  kernel créé : {kid[:8]}...")
    # Attendre que le kernel soit prêt
    for _ in range(30):
        k = _req("GET", f"/kernels/{kid}")
        if k.get("execution_state") == "idle":
            break
        time.sleep(1)
    return kid


def _delete_kernel(kid: str):
    try:
        _req("DELETE", f"/kernels/{kid}")
        print(f"  kernel supprimé : {kid[:8]}...")
    except Exception:
        pass


# ── WebSocket execution ───────────────────────────────────────────────────────

def _execute_on_kernel(kid: str, code: str, timeout: int = 600) -> list[str]:
    """Exécute `code` sur le kernel et retourne les lignes de sortie."""
    ws_url = (HUB
              .replace("https://", "wss://")
              .replace("http://",  "ws://")
              + f"/user/{USER}/api/kernels/{kid}/channels"
              + f"?token={TOKEN}")

    output_lines: list[str] = []
    msg_id = uuid.uuid4().hex
    done   = {"flag": False}

    def _on_open(ws):
        msg = {
            "header": {
                "msg_id":   msg_id,
                "username": USER,
                "session":  uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version":  "5.3",
            },
            "parent_header": {},
            "metadata":      {},
            "content": {
                "code":             code,
                "silent":           False,
                "store_history":    False,
                "user_expressions": {},
                "allow_stdin":      False,
                "stop_on_error":    True,
            },
            "channel": "shell",
        }
        ws.send(json.dumps(msg))

    def _on_message(ws, raw):
        msg = json.loads(raw)
        mtype  = msg.get("header", {}).get("msg_type", "")
        parent = msg.get("parent_header", {}).get("msg_id", "")
        if parent != msg_id:
            return

        if mtype == "stream":
            text = msg["content"].get("text", "").rstrip()
            if text:
                print("  SERVER |", text)
                output_lines.append(text)

        elif mtype in ("execute_result", "display_data"):
            text = msg["content"].get("data", {}).get("text/plain", "")
            if text:
                print("  SERVER |", text)
                output_lines.append(text)

        elif mtype == "error":
            tb = "\n".join(msg["content"].get("traceback", []))
            print("  ERROR  |", msg["content"].get("ename"),
                  msg["content"].get("evalue"))
            print(tb)
            output_lines.append(f"ERROR: {msg['content'].get('evalue')}")
            done["flag"] = True
            ws.close()

        elif mtype == "execute_reply":
            done["flag"] = True
            ws.close()

    def _on_error(ws, err):
        print("  WS ERROR:", err)
        done["flag"] = True

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
    )
    ws.run_forever(
        sslopt={"cert_reqs": ssl.CERT_NONE},
        ping_interval=30,
        ping_timeout=10,
    )
    return output_lines


# ── Download helpers ──────────────────────────────────────────────────────────

def _list_remote(remote_path: str) -> list[dict]:
    try:
        return _req("GET", f"/contents/work/dfi-quant/{remote_path}")["content"]
    except Exception:
        return []


def _download_file(remote_path: str, local_path: pathlib.Path):
    resp = _req("GET", f"/contents/work/dfi-quant/{remote_path}?type=file&format=base64")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(base64.b64decode(resp["content"]))


def _sync_partition(exchange: str, data_type: str, symbol: str,
                    year: str, month: str, day: str) -> bool:
    """Télécharge un partition day si pas déjà local. Retourne True si nouveau."""
    rel = (f"data/raw/exchange={exchange}/data_type={data_type}"
           f"/symbol={symbol}/year={year}/month={month}/day={day}")
    local_dir = RAW_ROOT / f"exchange={exchange}" / f"data_type={data_type}" \
                          / f"symbol={symbol}" / f"year={year}" \
                          / f"month={month}" / f"day={day}"

    success = local_dir / "_SUCCESS"
    if success.exists():
        return False   # déjà là

    files = _list_remote(rel)
    if not any(f["name"] == "_SUCCESS" for f in files):
        return False   # pas encore complète côté serveur

    for f in files:
        if f["name"].endswith(".parquet"):
            _download_file(f"{rel}/{f['name']}", local_dir / f["name"])
    success.touch()
    print(f"  ✓ téléchargé {symbol} {year}-{month}-{day} ({data_type})")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to",   dest="d_to",   required=True)
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--data-types", default="trades")
    args = ap.parse_args()

    symbols    = [s.strip() for s in args.symbols.split(",")]
    data_types = [d.strip() for d in args.data_types.split(",")]

    print(f"\n── Ingestion Tardis via JupyterHub ──")
    print(f"  période  : {args.d_from} → {args.d_to}")
    print(f"  symboles : {symbols}")
    print(f"  types    : {data_types}\n")

    # 1. Créer un kernel
    print("1. Création du kernel sur le serveur...")
    kid = _create_kernel()

    try:
        # 2. Lancer l'ingestion sur le serveur
        cmd = (
            f"import subprocess, sys\n"
            f"result = subprocess.run(\n"
            f"    [sys.executable, '-m', 'agents.tardis_ingestion.runner',\n"
            f"     '--from', '{args.d_from}', '--to', '{args.d_to}',\n"
            f"     '--symbols', '{args.symbols}',\n"
            f"     '--data-types', '{args.data_types}'],\n"
            f"    capture_output=True, text=True,\n"
            f"    cwd='/home/jovyan/work/dfi-quant'\n"
            f")\n"
            f"print(result.stdout)\n"
            f"if result.returncode != 0:\n"
            f"    print('STDERR:', result.stderr)\n"
        )
        print("2. Exécution de l'ingestion sur le serveur...")
        _execute_on_kernel(kid, cmd, timeout=600)

        # 3. Télécharger les fichiers résultants
        print("\n3. Téléchargement des partitions complètes...")
        import datetime as dt
        d0 = dt.date.fromisoformat(args.d_from)
        d1 = dt.date.fromisoformat(args.d_to)
        cur = d0
        downloaded = 0
        while cur <= d1:
            for sym in symbols:
                for dtp in data_types:
                    ok = _sync_partition(
                        "binance-futures", dtp, sym,
                        f"{cur:%Y}", f"{cur:%m}", f"{cur:%d}",
                    )
                    if ok:
                        downloaded += 1
            cur += dt.timedelta(days=1)

        print(f"\n  {downloaded} nouvelles partitions téléchargées.")

    finally:
        # 4. Supprimer le kernel
        print("\n4. Nettoyage...")
        _delete_kernel(kid)

    print("\n── Terminé ──\n")


if __name__ == "__main__":
    main()
