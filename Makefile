PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
MODEL_PATH := rf_model.joblib
DATA_FILE  := water_potability.csv

.PHONY: venv install
venv:
	python3 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
.PHONY: format lint security

format:
	$(PYTHON) -m black .
lint:
	$(PYTHON) -m flake8 --max-line-length=120 main.py model_pipeline.py

security:
	$(PYTHON) -m bandit -r main.py model_pipeline.py

.PHONY: prepare

prepare:
	$(PYTHON) main.py prepare --data $(DATA_FILE)
.PHONY: train train_and_save evaluate

train:
	$(PYTHON) main.py train --data $(DATA_FILE)

train_and_save:
	$(PYTHON) main.py train_and_save --data $(DATA_FILE) --model_path $(MODEL_PATH)

evaluate:
	$(PYTHON) main.py evaluate --data $(DATA_FILE) --model_path $(MODEL_PATH)
.PHONY: test

test:
	$(PYTHON) -m pytest

ci: format lint security test train

all: install ci 
api:
	.venv/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ======================
# Variables Docker
# ======================
IMAGE_NAME = sarah_mlops
DOCKERHUB_USER = sriserml
CONTAINER_NAME = sarah_mlops_container
PORT = 8000

# ======================
# Docker
# ======================
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p $(PORT):8000 $(IMAGE_NAME)

docker-login:
	docker login

docker-tag:
	docker tag $(IMAGE_NAME) $(DOCKERHUB_USER)/$(IMAGE_NAME):latest

docker-push:
	docker push $(DOCKERHUB_USER)/$(IMAGE_NAME):latest

docker-all: docker-build docker-tag docker-push
