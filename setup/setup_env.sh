#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="${1:-python}"
echo "Setting up virtual environment in .venv using '${PYTHON_CMD}'..."

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "Python command '${PYTHON_CMD}' not found. Please install Python 3.9+ and ensure it is on PATH." >&2
  exit 1
fi

VENV_DIR=".venv"
if [[ -d "${VENV_DIR}" ]]; then
  echo ".venv already exists. Skipping creation."
else
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
  echo "Virtual environment created at .venv."
fi

PIP_CMD=""
if [[ -x "${VENV_DIR}/bin/pip" ]]; then
  PIP_CMD="${VENV_DIR}/bin/pip"
elif [[ -x "${VENV_DIR}/Scripts/pip.exe" ]]; then
  PIP_CMD="${VENV_DIR}/Scripts/pip.exe"
fi

if [[ -n "${PIP_CMD}" ]]; then
  echo "Upgrading pip in the virtual environment..."
  "${PIP_CMD}" install --upgrade pip

  if [[ -f "requirements.txt" ]]; then
    echo "Installing dependencies from requirements.txt..."
    "${PIP_CMD}" install -r "requirements.txt"
  else
    echo "requirements.txt not found in the current directory; skipping dependency installation." >&2
  fi
else
  echo "pip executable not found in .venv. Please check the environment." >&2
fi

echo "Done. Activate the environment with:"
echo "    source .venv/bin/activate"

