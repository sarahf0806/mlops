# Image de base Python légère
FROM python:3.10-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans l'image
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Commande de lancement de l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
