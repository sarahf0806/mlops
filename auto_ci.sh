#!/usr/bin/env bash

echo "🚀 Auto CI lancé : toute modification dans le dossier va déclencher 'make ci'."
echo "Appuie sur CTRL+C pour arrêter."

while inotifywait -r -e modify,create,delete ./; do
    echo "🔄 Changement détecté, lancement de 'make ci'..."
    make ci
    echo "✅ Pipeline CI terminé. En attente de nouvelles modifications..."
done
