"""Two interchangeable ways for the UI to reach the backend. Every method
returns (data, error_message); exactly one of the two is None.

  LocalClient  runs the pipeline inside the Streamlit process (default; this is
               what makes a single free deployment possible).
  HttpClient   talks to the FastAPI server (set BACKEND_URL to use it).
"""
import io

import requests


class LocalClient:
    mode = "in-process"

    def __init__(self):
        # Imported here (not at module top) so HTTP mode never loads these.
        from backend import config
        from backend.services import (
            embedding_service,
            ingest_service,
            rag_service,
            system_info,
            vector_store,
        )
        from backend.services.ingest_service import IngestError
        from backend.services.llm_service import LLMError

        self._ingest = ingest_service
        self._rag = rag_service
        self._store = vector_store
        self._info = system_info
        self._errors = (IngestError, LLMError)

        embedding_service.warm_up()
        if config.DEMO_MODE:
            ingest_service.ensure_sample_indexed()

    def _call(self, function, *args):
        try:
            return function(*args), None
        except self._errors as exc:
            return None, str(exc)
        except Exception as exc:  # last resort: show a message instead of a traceback
            return None, f"Unexpected error ({type(exc).__name__}): {exc}"

    def documents(self):
        return self._call(self._store.list_documents)

    def upload(self, name, data):
        return self._call(self._ingest.ingest_pdf, name, io.BytesIO(data))

    def delete(self, name):
        return self._call(self._ingest.delete_document, name)

    def search(self, query, top_k, source):
        return self._call(
            lambda: {"query": query, "results": self._rag.retrieve(query, top_k, source)}
        )

    def ask(self, question, top_k, source):
        return self._call(self._rag.answer_question, question, top_k, source)

    def insight(self, kind, source):
        return self._call(self._rag.generate_insight, kind, source)

    def status(self):
        return self._call(self._info.system_info)


class HttpClient:
    mode = "http"

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _request(self, method, path, **kwargs):
        timeout = kwargs.pop("timeout", 600)
        try:
            response = requests.request(
                method, f"{self.base_url}{path}", timeout=timeout, **kwargs
            )
        except requests.ConnectionError:
            return None, (
                f"Cannot reach the backend at {self.base_url}. Start it with: "
                "uvicorn backend.main:app --port 8000"
            )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return None, str(detail)
        return response.json(), None

    def documents(self):
        return self._request("GET", "/documents")

    def upload(self, name, data):
        return self._request("POST", "/upload", files={"file": (name, data, "application/pdf")})

    def delete(self, name):
        from urllib.parse import quote

        return self._request("DELETE", f"/documents/{quote(name)}")

    def search(self, query, top_k, source):
        params = {"q": query, "top_k": top_k}
        if source:
            params["source"] = source
        return self._request("GET", "/search", params=params)

    def ask(self, question, top_k, source):
        body = {"question": question, "top_k": top_k, "source": source}
        return self._request("POST", "/ask", json=body)

    def insight(self, kind, source):
        return self._request("POST", "/insights", json={"kind": kind, "source": source})

    def status(self):
        return self._request("GET", "/db-info")
