from elasticsearch import Elasticsearch
from datetime import datetime

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "mlflow-metrics"


def log_to_elasticsearch(run_id: str, metric_name: str, value: float):
    doc = {
        "run_id": run_id,
        "metric": metric_name,
        "value": value,
        "timestamp": datetime.utcnow(),
    }
    es.index(index=INDEX_NAME, document=doc)
