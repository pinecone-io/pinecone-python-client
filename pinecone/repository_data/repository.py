import logging
import json
from typing import Optional, Dict, List, Any, Tuple, Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from multiprocessing import cpu_count
from pinecone.core.openapi.repository_data import API_VERSION
from .models.document import Document

logger = logging.getLogger(__name__)


def _ensure_https_host(host: str) -> str:
    """
    Normalizes the host value to include scheme and no trailing slash.
    Accepts: "kb.example.com", "https://kb.example.com/", "http://..."
    Returns: "https://kb.example.com"
    """
    host = (host or "").strip()
    if not host:
        raise ValueError("host must be provided (e.g., 'kb.your-company.com').")
    if not host.startswith(("http://", "https://")):
        host = "https://" + host
    # strip single trailing slash
    if host.endswith("/"):
        host = host[:-1]
    return host


class HTTPError(Exception):
    """Rich HTTP error including status code and server payload (if any)."""

    def __init__(self, status_code: int, message: str, payload: Optional[dict] = None):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.payload = payload or {}


class Repository:
    """
    A client for interacting with the Pinecone Knowledge Base Data Plane (Documents).
    Uses `requests` directly, with retries and sane defaults.

    Methods return plain `dict` responses parsed from JSON.
    """

    def __init__(
        self,
        api_key: str,
        host: str,
        pool_threads: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        openapi_config=None,  # kept for backward compat; unused
        echo: bool = False,
        **kwargs,
    ):
        self._api_key = api_key
        self._base_url = _ensure_https_host(host)
        self._echo = echo  # store the flag

        # Connection pool sizing
        self._pool_threads = 5 * cpu_count() if pool_threads is None else pool_threads
        pool_maxsize = kwargs.get("connection_pool_maxsize", self._pool_threads)

        # Timeouts (connect, read). Allow overrides via kwargs
        # e.g., timeout=(3.05, 30)
        self._timeout: Tuple[float, float] = kwargs.get("timeout", (5.0, 60.0))

        # Retries: conservative defaults; override via kwargs["retries"]
        retries = kwargs.get(
            "retries",
            Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "POST", "DELETE"]),
                raise_on_status=False,
            ),
        )

        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self._pool_threads, pool_maxsize=pool_maxsize, max_retries=retries
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._default_headers = {
            "Api-Key": self._api_key,
            "Accept": "application/json",
            "x-pinecone-api-version": API_VERSION,
            # Content-Type set per request when needed
        }
        if additional_headers:
            self._default_headers.update(additional_headers)

    # -----------------------
    # Internal request helper
    # -----------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        echo: Optional[bool] = None,
    ) -> Dict:
        url = urljoin(self._base_url + "/", path.lstrip("/"))
        hdrs = dict(self._default_headers)
        if headers:
            hdrs.update(headers)
        if json_body is not None:
            hdrs.setdefault("Content-Type", "application/json")

        logger.debug("HTTP %s %s params=%s json=%s", method, url, params, json_body)

        # decide whether to echo this call
        do_echo = self._echo if echo is None else echo
        if do_echo:
            print("----- HTTP Request -----")
            print(f"{method} {url}")
            if params:
                print("Params:", params)

            safe_headers = dict(hdrs)
            for k, v in hdrs.items():
                print(f"checking........... {k}: {v}")
                if k.lower() == "api-key":
                    masked = (v[:5] + "...") if isinstance(v, str) and len(v) > 5 else "..."
                    safe_headers[k] = masked
                else:
                    safe_headers[k] = v

            print("Headers:", safe_headers)
            if json_body is not None:
                print("Body:", json.dumps(json_body, indent=2))
            print("------------------------")

        resp = self._session.request(
            method=method,
            url=url,
            headers=hdrs,
            params=params,
            json=json_body,
            timeout=self._timeout,
        )

        # Try to parse JSON payload (even on errors) for better messages
        payload: Optional[dict]
        try:
            payload = resp.json() if resp.content else None
        except json.JSONDecodeError:
            payload = None

        if not (200 <= resp.status_code < 300):
            msg = payload.get("message") if isinstance(payload, dict) else resp.text
            raise HTTPError(resp.status_code, msg or "HTTP request failed", payload)

        if payload is None:
            return {}
        return payload

    # -------------
    # API methods
    # -------------
    def upsert(self, namespace: str, document: Union[Dict[str, Any], Document], **kwargs) -> Dict:
        """
        POST /namespaces/{namespace}/documents/upsert
        Returns UpsertDocumentResponse as dict.
        """
        if isinstance(document, Document):
            json_body = document.to_dict()
        elif isinstance(document, dict):
            json_body = document
        else:
            raise TypeError("document must be a dict or Document.")

        path = f"/namespaces/{namespace}/documents/upsert"
        return self._request("POST", path, json_body=json_body, **kwargs)

    def fetch(self, namespace: str, document_id: str, **kwargs) -> Document:
        """
        GET /namespaces/{namespace}/documents/{document_id}
        Returns GetDocumentResponse as dict.
        """
        path = f"/namespaces/{namespace}/documents/{document_id}"
        payload = self._request("GET", path, **kwargs)

        if not isinstance(payload, dict):
            raise ValueError("Unexpected fetch-documents response format, expected dict response")

        doc = payload.get("document")
        if not doc or not isinstance(doc, dict):
            raise ValueError(
                "Unexpected fetch-documents response format, expected 'document' key of type dict"
            )

        return Document.from_api(doc)

    def list(self, namespace: str, **kwargs) -> List[Document]:
        """
        GET /namespaces/{namespace}/documents
        Returns ListDocumentsResponse as dict.
        """
        path = f"/namespaces/{namespace}/documents"
        # Spec does not define query params, but keep hook if server adds (e.g., pagination).
        params = kwargs.get("params")
        payload = self._request("GET", path, params=params, **kwargs)

        candidates: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            raise ValueError("Unexpected list-documents response format, expected dict response")

        docs = payload.get("documents")
        if not docs or not isinstance(docs, list):
            raise ValueError(
                "Unexpected list-documents response format, expected 'documents' key of type list"
            )

        candidates = docs

        return [Document.from_api(doc) for doc in candidates]

    def delete(self, namespace: str, document_id: str, **kwargs):
        """
        DELETE /namespaces/{namespace}/documents/{document_id}
        Returns DeleteDocumentResponse as dict.
        """
        path = f"/namespaces/{namespace}/documents/{document_id}"
        return self._request("DELETE", path, **kwargs)
