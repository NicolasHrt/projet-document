#!/usr/bin/env bash
# Lance le projet complet : environnement Python, dépendances, backend + RAG.
# Usage : ./start.sh
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

echo "=== Traitement de courrier administratif ==="

# 1. Python 3 disponible ?
if ! command -v python3 >/dev/null; then
  echo "❌ python3 introuvable. Installe Python 3.10+ puis relance."
  exit 1
fi

# 2. Environnement virtuel
if [ ! -d .venv ]; then
  echo "→ Création de l'environnement virtuel (.venv)…"
  python3 -m venv .venv
fi

# 3. Dépendances (rapide si déjà installées)
echo "→ Installation des dépendances (la première fois : plusieurs minutes, PyTorch est volumineux)…"
.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt

# 4. Fichier .env / clé API
if [ ! -f .env ]; then
  echo "ANTHROPIC_API_KEY=" > .env
  echo "⚠️  Fichier .env créé. Mets-y ta clé Anthropic (console.anthropic.com) :"
  echo "    ANTHROPIC_API_KEY=sk-ant-..."
fi
if ! grep -q "ANTHROPIC_API_KEY=sk-" .env 2>/dev/null; then
  echo "⚠️  Aucune clé Anthropic détectée dans .env — l'analyse IA et les brouillons ne marcheront pas."
  echo "    (l'extraction PDF et la recherche RAG fonctionnent quand même)"
fi

# 5. Lancement
echo ""
echo "→ Démarrage du serveur sur http://localhost:${PORT}"
echo "  • Interface     : ouvre index.html dans ton navigateur"
echo "  • Doc API       : http://localhost:${PORT}/docs"
echo "  • Premier envoi : le modèle d'embeddings (~1 Go) se télécharge automatiquement"
echo "  Ctrl+C pour arrêter."
echo ""

# Ouvre l'interface automatiquement sur macOS
if [ "$(uname)" = "Darwin" ]; then
  (sleep 2 && open index.html) &
fi

exec .venv/bin/uvicorn api.main:app --reload --port "${PORT}"
