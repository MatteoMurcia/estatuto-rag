#!/usr/bin/env bash
# Reproduce en local los mismos pasos que .github/workflows/ci.yml.
# Si cambias la CI, cambia también este script (y al revés).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$1"; }

step "uv sync --locked"
uv sync --locked

step "Lint (ruff check)"
uv run ruff check .

step "Formato (ruff format --check)"
uv run ruff format --check .

step "Tipos (mypy)"
uv run mypy

step "Tests (pytest) — necesita la BD: docker compose up -d db"
uv run pytest --cov=estatuto_rag --cov-report=term-missing

step "Secretos (gitleaks)"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks git --no-banner .
else
  echo "gitleaks no está instalado: se salta en local (la CI sí lo ejecuta)"
fi

printf '\nTodo verde.\n'
