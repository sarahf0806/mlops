from datetime import datetime
from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:9200"
INDEX_NAME = "mlops-metrics"

es = Elasticsearch(
    ES_HOST,
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
)

def log_to_elasticsearch(run_id: str, metric_name: str, value: float):
    """
    Envoie une métrique MLflow vers Elasticsearch
    """
    doc = {
        "run_id": run_id,
        "metric": metric_name,
        "value": float(value),
        "timestamp": datetime.utcnow(),
        "source": "mlops-pipeline",
    }

    es.index(index=INDEX_NAME, document=doc)
