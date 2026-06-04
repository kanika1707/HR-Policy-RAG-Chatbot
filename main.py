"""
Setup & launcher for HR Policy RAG Chatbot.
Run this once to set everything up, then again any time to launch the app.

    python main.py
"""

import subprocess
import sys
import shutil
from pathlib import Path

ROOT         = Path(__file__).parent
VENV_DIR     = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"
ENV_FILE     = ROOT / ".env"
ENV_EXAMPLE  = ROOT / ".env.example"
APP          = ROOT / "streamlit_app.py"

# ── Platform-aware paths inside the venv ────────────────────────────────────
if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"


def banner(step: str, msg: str):
    print(f"\n{'─' * 55}")
    print(f"  {step}  {msg}")
    print(f"{'─' * 55}")


def run(cmd: list, **kwargs):
    """Run a command and exit on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n❌  Command failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(1)


def create_venv():
    banner("[ 1/4 ]", "Virtual environment")
    if VENV_DIR.exists():
        print("  ✔  venv already exists — skipping creation.")
    else:
        print("  Creating venv …")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("  ✔  venv created.")


def install_requirements():
    banner("[ 2/4 ]", "Installing packages")
    print("  Upgrading pip …")
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    print("  Installing requirements.txt … (this may take a few minutes on first run)")
    run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS), "-q"])
    print("  ✔  All packages installed.")


def setup_env():
    banner("[ 3/4 ]", "Environment variables (.env)")
    if ENV_FILE.exists():
        print("  ✔  .env already exists — skipping.")
        return

    # Copy template
    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)

    print("  You need a free Groq API key to run this app.")
    print("  Sign up at: https://console.groq.com  (no credit card needed)\n")
    api_key = input("  Paste your GROQ_API_KEY here: ").strip()

    if not api_key:
        print("  ⚠  No key entered. Edit rag-chatbot/.env manually before running the app.")
    else:
        ENV_FILE.write_text(f"GROQ_API_KEY={api_key}\n")
        print("  ✔  .env file created.")


def launch_app():
    banner("[ 4/4 ]", "Launching Streamlit app")
    print("  App will open in your browser automatically.")
    print("  Press Ctrl+C here to stop the server.\n")
    run([str(VENV_PYTHON), "-m", "streamlit", "run", str(APP)])


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   HR Policy RAG Chatbot — Setup & Run   ║")
    print("╚══════════════════════════════════════════╝")

    create_venv()
    install_requirements()
    setup_env()
    launch_app()
