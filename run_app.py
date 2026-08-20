"""
Single-Port Production Launcher for Medflow Medical RAG Engine (Port 7860)
FastAPI serves the SPA frontend (index.html), health probe (/health), and API (/api/v1/*).
"""

import os
import sys
import time
import socket
import threading
import webbrowser
import urllib.request
import json
import subprocess

# Ensure ASCII/UTF-8 output safety on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def kill_stale_process_on_port(port: int):
    try:
        cmd = f"$conns = Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue; if ($conns) {{ foreach ($p in ($conns | Select-Object -ExpandProperty OwningProcess -Unique)) {{ if ($p -gt 0) {{ Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }} }} }}"
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd], capture_output=True, text=True)
        time.sleep(1.5)
    except Exception:
        pass

def check_existing_health(port: int = 7860) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if (data.get("status") == "healthy" or data.get("ready") is True) and data.get("active_chunks", 0) >= 1900:
                    return True
    except Exception:
        pass
    return False

def open_browser_when_ready(port: int = 7860, timeout: int = 60):
    start_time = time.time()
    url = f"http://127.0.0.1:{port}"
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    if data.get("ready") is True or data.get("status") == "healthy":
                        print("\n" + "=" * 65)
                        print("  SUCCESS! Medflow20 Core Engine & Web Interface Ready!")
                        print(f"  Application URL: {url}")
                        print("=" * 65 + "\n")
                        time.sleep(0.5)
                        webbrowser.open(url)
                        return
        except Exception:
            pass
        time.sleep(1)
    print("\n[WARNING] Backend health probe timed out. Opening browser anyway...")
    webbrowser.open(url)

def main():
    root_dir = get_project_root()
    os.chdir(root_dir)
    
    print("=" * 65)
    print("  Medflow Medical RAG Assistant (Medflow20 Core Engine)")
    print("  Local Host URL: http://127.0.0.1:7860")
    print("=" * 65)

    # 1. Inspect port 7860
    if is_port_in_use(7860):
        print("[Launcher] Port 7860 is currently occupied. Testing health...")
        if check_existing_health(7860):
            print("[Launcher] Active, healthy Medflow engine detected on port 7860!")
            print("[Launcher] Opening Web UI...")
            webbrowser.open("http://127.0.0.1:7860")
            sys.exit(0)
        else:
            print("[Launcher] Terminating stale process on port 7860...")
            kill_stale_process_on_port(7860)

    # 2. Environment setup
    medflow20_dir = os.path.join(root_dir, "medflow20")
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    if medflow20_dir not in sys.path:
        sys.path.insert(0, medflow20_dir)

    os.environ["PYTHONPATH"] = root_dir + os.pathsep + medflow20_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PORT"] = "7860"

    print("\n[1/2] Initializing Medflow20 Core RAG Engine & FastAPI on http://127.0.0.1:7860...")
    print("[2/2] Waiting for Medflow20 vector store & embeddings to load...")

    # 3. Start readiness probe thread to open browser when ready
    probe_thread = threading.Thread(target=open_browser_when_ready, args=(7860, 60), daemon=True)
    probe_thread.start()

    # 4. Run uvicorn in main thread to keep process alive and stream stdout/stderr
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=7860, log_level="info")

if __name__ == "__main__":
    main()
