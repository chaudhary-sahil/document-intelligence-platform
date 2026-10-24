"""One place that reports what the app is currently running on."""
from backend import config
from backend.services import embedding_service, llm_service, vector_store


def system_info():
    info = vector_store.stats()
    info.update(
        {
            "embedding_backend": embedding_service.backend_name(),
            "llm": llm_service.describe(),
            "demo_mode": config.DEMO_MODE,
            "max_upload_mb": config.MAX_UPLOAD_MB,
        }
    )
    return info
