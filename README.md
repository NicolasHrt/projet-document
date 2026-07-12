# Traitement de courrier administratif

App de réception de courriers administratifs (photo, scan, PDF, texte) : numérisation, classification IA, et génération de brouillons de réponse ancrés sur une base documentaire (RAG).

## Démarrage rapide

```bash
./start.sh
```

Le script crée l'environnement Python, installe les dépendances, prépare le fichier `.env` et lance le serveur sur http://localhost:8000. L'interface (`index.html`) s'ouvre automatiquement sur macOS.

**Prérequis :** Python 3.10+ et une clé API Anthropic (console.anthropic.com) à mettre dans `.env` :

```
ANTHROPIC_API_KEY=sk-ant-...
```

Au premier courrier envoyé, le modèle d'embeddings (~1 Go) se télécharge — c'est normal que ce soit long une fois.

## Utilisation

1. **Base documentaire** : indexe tes PDF de référence (modèles de réponses, courriers passés, pièces de dossiers avec leur référence). La qualité des brouillons en dépend.
2. **Boîte de réception** : dépose un courrier reçu (photo/PDF) ou colle son texte → il est numérisé, classé (catégorie, urgence) et sa référence de dossier est détectée.
3. Ouvre le courrier → **Générer le brouillon** → édite, copie, passe le statut à « traité ».
4. **Statistiques** : compteurs par catégorie et statut.

## Architecture

```
index.html          Front vanilla JS (3 onglets)
api/main.py         Endpoints FastAPI
api/pipeline.py     IA : numérisation (vision), classification, brouillons (Claude)
api/rag.py          Base vectorielle ChromaDB + embeddings locaux (sentence-transformers)
api/db.py           SQLite (courriers, catégories, statuts)
```

Les données restent locales : `app.db` (courriers) et `chroma_db/` (index vectoriel), tous deux gitignorés. Seuls les extraits pertinents d'un courrier transitent par l'API Anthropic.
