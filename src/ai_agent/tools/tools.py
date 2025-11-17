# tools.py
from typing import Dict, Any, List
from tools_base import tool

import os
import json
import subprocess
import uuid
import glob
import requests


# =====================================================================
# 1. FILE SYSTEM TOOLS
# =====================================================================

@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (overwrites)."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if "/" in path else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "OK"


@tool
def append_file(path: str, content: str) -> str:
    """Append content to a file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    return "OK"


@tool
def list_files(pattern: str = "**/*") -> List[str]:
    """List files using a glob pattern."""
    return glob.glob(pattern, recursive=True)


@tool
def make_dir(path: str) -> str:
    """Create a directory."""
    os.makedirs(path, exist_ok=True)
    return "OK"


@tool
def delete_file(path: str) -> str:
    """Delete a file if exists."""
    if os.path.exists(path):
        os.remove(path)
    return "OK"


# =====================================================================
# 2. CODE EXECUTION TOOLS
# =====================================================================

@tool
def run_bash(cmd: str) -> str:
    """Run a bash command."""
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def run_python(code: str) -> str:
    """Execute Python code in a sandbox subprocess."""
    out = subprocess.run(["python3", "-c", code],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def run_node(code: str) -> str:
    """Execute JavaScript (Node) code."""
    out = subprocess.run(["node", "-e", code], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


# =====================================================================
# 3. PACKAGE MANAGEMENT TOOLS
# =====================================================================

@tool
def install_python_package(package: str) -> str:
    """Install a Python package using pip."""
    out = subprocess.run(f"pip install {package}", shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def install_npm_package(package: str) -> str:
    """Install an NPM package."""
    out = subprocess.run(f"npm install {package}", shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


# =====================================================================
# 4. GIT TOOLS
# =====================================================================

@tool
def git_clone(repo_url: str, destination: str = ".") -> str:
    """Clone a Git repository."""
    out = subprocess.run(
        f"git clone {repo_url} {destination}",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return out.stdout + "\n" + out.stderr


@tool
def git_status() -> str:
    """Git status."""
    out = subprocess.run("git status", shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def git_pull() -> str:
    """Git pull."""
    out = subprocess.run("git pull", shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def git_push(msg: str) -> str:
    """Git commit + push."""
    subprocess.run("git add .", shell=True)
    subprocess.run(f'git commit -m "{msg}"', shell=True)
    out = subprocess.run("git push", shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def git_commit(msg: str) -> str:
    """Git commit only."""
    out = subprocess.run(f'git commit -am "{msg}"', shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


# =====================================================================
# 5. HTTP TOOLS
# =====================================================================

@tool
def http_get(url: str) -> Dict[str, Any]:
    """HTTP GET request."""
    r = requests.get(url)
    return {"status": r.status_code, "body": r.text}


@tool
def http_post(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """HTTP POST with JSON body."""
    r = requests.post(url, json=data)
    return {"status": r.status_code, "body": r.text}


# =====================================================================
# 6. SEARCH TOOLS
# =====================================================================

@tool
def search_in_file(path: str, query: str) -> List[str]:
    """Search for a query in a single file."""
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines(), start=1):
            if query in line:
                lines.append(f"{i}: {line.strip()}")
    return lines


@tool
def search_in_project(query: str, pattern: str = "**/*.py") -> Dict[str, List[str]]:
    """Search across project files."""
    results = {}
    for file in glob.glob(pattern, recursive=True):
        matches = search_in_file(file, query)
        if matches:
            results[file] = matches
    return results


# =====================================================================
# 7. TESTING TOOLS
# =====================================================================

@tool
def run_tests_py() -> str:
    """Run Python tests (pytest)."""
    out = subprocess.run("pytest -q", shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


@tool
def run_tests_js() -> str:
    """Run JS tests (npm test)."""
    out = subprocess.run("npm test", shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    return out.stdout + "\n" + out.stderr


# =====================================================================
# 8. UTILITY TOOLS
# =====================================================================

@tool
def format_code(language: str, code: str) -> str:
    """Format code using tools like black, prettier, etc."""
    if language.lower() == "python":
        return subprocess.run(
            "black -q -",
            input=code,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ).stdout

    if language.lower() in ["javascript", "typescript", "js", "ts"]:
        return subprocess.run(
            "npx prettier --stdin-filepath temp.js",
            input=code,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ).stdout

    return code


@tool
def generate_uuid() -> str:
    """Generate a UUID."""
    return str(uuid.uuid4())
