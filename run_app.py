"""
Compatibility launcher for the single-port Medflow20 application.
FastAPI serves both the website and API on port 7860.
"""

import os
import sys
import subprocess
import time
import socket

# Ensure ASCII/UTF-8 output safety on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def kill_process_on_port(port: int):
    try:
        cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {{ if ($_) {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }} }}"
        subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    except Exception:
        pass

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prefer local virtualenv python if present
    local_venv_python = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    if os.path.exists(local_venv_python):
        python_exe = local_venv_python
    else:
        python_exe = sys.executable

    print("=" * 65)
    print(">>> Starting Medflow Medical RAG End-to-End Application (Medflow20)...")
    print(f"Using Python executable: {python_exe}")
    print("=" * 65)

    # 1. Never replace another process on the primary application port.
    for port in [7860]:
        if is_port_in_use(port):
            print(f"[ERROR] Port {port} is already in use. Start Medflow with start_medflow.bat to reuse a healthy instance.")
            sys.exit(1)

    env = os.environ.copy()
    medflow20_dir = os.path.join(root_dir, "medflow20")
    env["PYTHONPATH"] = root_dir + os.pathsep + medflow20_dir + os.pathsep + env.get("PYTHONPATH", "")
    # The project ships its RAG models locally. Avoid startup delays and failures when
    # the development environment has restricted outbound network access.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 2. Start the complete FastAPI application on the single local port.
    print("\n[1/1] Launching Medflow20 on http://127.0.0.1:7860 ...")
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "7860"],
        cwd=root_dir,
        env=env
    )

    # Wait for FastAPI and the Medflow20 corpus to be fully ready.
    print("[Launcher] Waiting for Medflow20 RAG Engine & FastAPI to initialize on port 7860...")
    max_wait = 180
    start_time = time.time()
    backend_ready = False
    import urllib.request
    import json

    while time.time() - start_time < max_wait:
        if backend_proc.poll() is not None:
            print(f"\n[ERROR] FastAPI Backend process exited unexpectedly! (Exit code: {backend_proc.returncode})")
            sys.exit(1)
        
        try:
            req = urllib.request.Request("http://127.0.0.1:7860/health")
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    backend_ready = True
                    break
        except Exception:
            pass
        time.sleep(1)

    if not backend_ready:
        print("\n[ERROR] Medflow20 backend failed to become ready within timeout.")
        backend_proc.terminate()
        sys.exit(1)

    print("\n[SUCCESS] Medflow Application Services Initialized Successfully!")
    print("------------------------------------------------------------")
    print("  Medflow UI:       http://127.0.0.1:7860")
    print("  Backend API Docs: http://127.0.0.1:7860/docs")
    print("------------------------------------------------------------")
    print("Press Ctrl+C to stop all processes.\n")

    try:
        while True:
            if backend_proc.poll() is not None:
                print("\n[WARNING] FastAPI Backend process stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Medflow services...")
    finally:
        backend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
