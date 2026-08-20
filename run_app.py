"""
Medflow Medical RAG Application Launcher
Starts both FastAPI Backend (Port 8000) and Frontend HTTP Server (Port 3000) concurrently.
Guarantees clean port binding and displays real-time backend startup feedback.
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

    # 1. Clean up orphan background processes on ports 8000 and 3000
    for port in [8000, 3000]:
        if is_port_in_use(port):
            print(f"[Launcher] Port {port} is in use by a previous process. Cleaning up...")
            kill_process_on_port(port)
            time.sleep(1)

    env = os.environ.copy()
    medflow20_dir = os.path.join(root_dir, "medflow20")
    env["PYTHONPATH"] = root_dir + os.pathsep + medflow20_dir + os.pathsep + env.get("PYTHONPATH", "")
    # The project ships its RAG models locally. Avoid startup delays and failures when
    # the development environment has restricted outbound network access.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 2. Start FastAPI Backend on Port 8000
    print("\n[1/2] Launching FastAPI Backend (Medflow20 Engine) on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=root_dir,
        env=env
    )

    # 3. Start Frontend HTTP Server on Port 3000
    print("[2/2] Launching Frontend HTTP Server on http://127.0.0.1:3000 ...")
    frontend_proc = subprocess.Popen(
        [python_exe, "-m", "http.server", "3000", "--bind", "127.0.0.1"],
        cwd=root_dir
    )

    # Wait for FastAPI backend to be fully bound and ready on port 8000
    print("[Launcher] Waiting for Medflow20 RAG Engine & FastAPI to initialize on port 8000...")
    max_wait = 45
    start_time = time.time()
    backend_ready = False
    import urllib.request
    import json

    while time.time() - start_time < max_wait:
        if backend_proc.poll() is not None:
            print(f"\n[ERROR] FastAPI Backend process exited unexpectedly! (Exit code: {backend_proc.returncode})")
            frontend_proc.terminate()
            sys.exit(1)
        
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/health")
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
        frontend_proc.terminate()
        sys.exit(1)

    print("\n[SUCCESS] Medflow Application Services Initialized Successfully!")
    print("------------------------------------------------------------")
    print("  Frontend UI:      http://127.0.0.1:3000/index.html")
    print("  Backend API Docs: http://127.0.0.1:8000/docs")
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
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
