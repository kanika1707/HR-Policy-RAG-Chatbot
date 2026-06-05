"""
main.py — One-command setup and launcher for the HR Policy RAG Chatbot.

Anyone who forks this project only needs to run:
    python main.py

What this script does, in order:
  [1/4]  Creates a Python virtual environment (venv/) in the project folder
  [2/4]  Installs all packages listed in requirements.txt into that venv
  [3/4]  Creates a .env file with the user's Groq API key
  [4/4]  Launches the Streamlit app in the browser

Why use a virtual environment?
  A venv is an isolated Python installation just for this project.
  It prevents package version conflicts with other projects on the machine.
  All packages are installed inside venv/ — nothing is changed globally.

Why not just activate the venv?
  Activating a venv (e.g. venv\\Scripts\\activate) only works in the current
  shell session and cannot be done from inside a Python script.
  Instead, we call the venv's Python executable directly using its full path
  (venv/Scripts/python.exe on Windows, venv/bin/python on macOS/Linux).
  This achieves the same result — all commands run inside the isolated env.
"""

import subprocess   # used to run shell commands (pip install, streamlit run, etc.)
import sys          # gives us the path to the current Python interpreter
import shutil       # used to copy .env.example → .env
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent           # rag-chatbot/ folder
VENV_DIR     = ROOT / "venv"                  # virtual environment folder
REQUIREMENTS = ROOT / "requirements.txt"       # package list
ENV_FILE     = ROOT / ".env"                   # actual env file (contains API key — NOT in git)
ENV_EXAMPLE  = ROOT / ".env.example"          # template file (safe to share — no real key)
APP          = ROOT / "streamlit_app.py"       # the Streamlit UI file

# ── Platform-aware venv Python path ───────────────────────────────────────────
# On Windows the venv Python lives under Scripts/
# On macOS / Linux it lives under bin/
if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"


def banner(step: str, msg: str):
    """Print a formatted section header so the user can follow progress."""
    print(f"\n{'─' * 55}")
    print(f"  {step}  {msg}")
    print(f"{'─' * 55}")


def run(cmd: list, **kwargs):
    """
    Run an external command and exit the script immediately if it fails.

    Args:
        cmd    : list of strings, e.g. ["python", "-m", "pip", "install", "streamlit"]
        kwargs : forwarded to subprocess.run (e.g. cwd, env)
    """
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        # Non-zero exit code means the command failed — stop everything
        print(f"\n❌  Command failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(1)


def create_venv():
    """
    Step 1 — Create the virtual environment.

    Uses the current Python interpreter (sys.executable) to create a new
    isolated environment at VENV_DIR.  If venv/ already exists we skip
    creation so repeated runs of main.py don't overwrite the installed packages.
    """
    banner("[ 1/4 ]", "Virtual environment")
    if VENV_DIR.exists():
        # venv already set up from a previous run — nothing to do
        print("  ✔  venv already exists — skipping creation.")
    else:
        print("  Creating venv …")
        # python -m venv venv/  →  creates the isolated environment
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("  ✔  venv created.")


def install_requirements():
    """
    Step 2 — Install all project dependencies into the venv.

    We call the venv's own Python (VENV_PYTHON) so packages land inside
    the venv, not in the global Python installation.

    -q flag suppresses verbose pip output so the terminal stays readable.
    """
    banner("[ 2/4 ]", "Installing packages")

    # Always upgrade pip first to avoid warnings about outdated pip
    print("  Upgrading pip …")
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("  Installing requirements.txt … (this may take a few minutes on first run)")
    # -r requirements.txt tells pip to install every package listed in the file
    run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS), "-q"])

    print("  ✔  All packages installed.")


def setup_env():
    """
    Step 3 — Create the .env file with the user's Groq API key.

    .env is listed in .gitignore so it is NEVER committed to GitHub.
    If .env already exists (e.g. the user ran main.py before), we skip
    this step entirely so we don't overwrite their key.

    Flow:
      1. Copy .env.example → .env  (so the file exists with the right key name)
      2. Ask the user to paste their Groq API key
      3. Write the key into .env
    """
    banner("[ 3/4 ]", "Environment variables (.env)")

    if ENV_FILE.exists():
        # Key was already configured in a previous run
        print("  ✔  .env already exists — skipping.")
        return

    # Copy the template so .env has the correct structure
    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)

    print("  You need a free Groq API key to run this app.")
    print("  Sign up at: https://console.groq.com  (no credit card needed)\n")
    api_key = input("  Paste your GROQ_API_KEY here: ").strip()

    if not api_key:
        # User pressed Enter without typing — warn them to edit the file manually
        print("  ⚠  No key entered. Edit rag-chatbot/.env manually before running the app.")
    else:
        # Write the key into .env in the format: GROQ_API_KEY=gsk_...
        ENV_FILE.write_text(f"GROQ_API_KEY={api_key}\n")
        print("  ✔  .env file created.")


def launch_app():
    """
    Step 4 — Start the Streamlit web server using the venv's Python.

    Streamlit will:
      1. Print a local URL (e.g. http://localhost:8501)
      2. Automatically open that URL in the default browser

    The server runs until the user presses Ctrl+C in this terminal.
    """
    banner("[ 4/4 ]", "Launching Streamlit app")
    print("  App will open in your browser automatically.")
    print("  Press Ctrl+C here to stop the server.\n")

    # python -m streamlit run streamlit_app.py
    run([str(VENV_PYTHON), "-m", "streamlit", "run", str(APP)])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   HR Policy RAG Chatbot — Setup & Run   ║")
    print("╚══════════════════════════════════════════╝")

    create_venv()           # Step 1
    install_requirements()  # Step 2
    setup_env()             # Step 3
    launch_app()            # Step 4
