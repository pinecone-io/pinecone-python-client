"""Backwards-compatibility shim for the legacy ``Assistant`` plugin class.

Re-exports the ``Assistant``, ``Evaluation``, and ``Metrics`` classes that
used to live in the pre-rewrite ``pinecone-plugin-assistant`` distribution.
Preserved to keep pre-rewrite callers working. New code should use the
Pinecone client assistants namespace directly.

:meta private:
"""

import os
import time
from typing import Any, List, Optional
from urllib.parse import urljoin

from pinecone_plugin_interface import PineconePlugin

from pinecone_plugins.assistant.models.assistant_model import (
    HOST_SUFFIX,
    AssistantModel,
)
from pinecone_plugins.assistant.models.evaluation_responses import AlignmentResponse
from pinecone_plugins.assistant.models.list_assistants_response import ListAssistantsResponse


class Metrics:
    """Evaluation metrics API helper."""

    def __init__(self, metrics_api: Any) -> None:
        self._metrics_api = metrics_api

    def alignment(
        self,
        question: str,
        answer: str,
        ground_truth_answer: str,
    ) -> AlignmentResponse:
        """Compute alignment between an answer and a ground-truth answer."""
        request = {
            "question": question,
            "answer": answer,
            "ground_truth_answer": ground_truth_answer,
        }
        return AlignmentResponse.from_openapi(
            self._metrics_api.metrics_alignment(alignment_request=request)
        )


class Evaluation:
    """Evaluation sub-resource initialised from the data-plane host."""

    def __init__(self, client_builder: Any) -> None:
        host = os.getenv(
            "PINECONE_PLUGIN_ASSISTANT_DATA_HOST", "https://prod-1-data.ke.pinecone.io"
        )
        self.host = urljoin(host, HOST_SUFFIX)
        metrics_api = client_builder(host=self.host)
        self.metrics = Metrics(metrics_api)


class Assistant(PineconePlugin):
    """Legacy ``Assistant`` plugin class from the pre-rewrite plugin distribution.

    Provides CRUD operations for assistants and is installed on the
    ``Pinecone.assistant`` namespace via the plugin interface.
    """

    def __init__(self, config: Any, client_builder: Any) -> None:
        self.config = config

        host = os.getenv("PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST", "https://api.pinecone.io")
        self.host = urljoin(host, HOST_SUFFIX)

        self._assistant_control_api = client_builder(host=self.host)
        self._client_builder = client_builder
        self.evaluation = Evaluation(client_builder=self._client_builder)

    def create_assistant(
        self,
        assistant_name: str,
        instructions: Optional[str] = None,
        metadata: dict[str, Any] = {},  # noqa: B006
        region: Optional[str] = "us",
        timeout: Optional[int] = None,
    ) -> AssistantModel:
        """Create a new assistant and wait until it is Ready."""
        if region not in ["us", "eu"]:
            raise ValueError("Region must be either 'us' or 'eu'")

        create_request = {
            "name": assistant_name,
            "instructions": instructions,
            "metadata": metadata,
            "region": region,
        }
        assistant = self._assistant_control_api.create_assistant(
            create_assistant_request=create_request
        )

        if timeout == -1:
            return AssistantModel(
                assistant=assistant,
                client_builder=self._client_builder,
                config=self.config,
            )

        if timeout is None:
            while not assistant.status == "Ready":  # noqa: SIM201
                time.sleep(0.5)
                assistant = self.describe_assistant(assistant_name)
        else:
            while not assistant.status == "Ready" and timeout >= 0:  # noqa: SIM201
                time.sleep(0.5)
                timeout -= 0.5
                assistant = self.describe_assistant(assistant_name)

        if timeout and timeout < 0:
            raise TimeoutError(
                "Please call the describe_assistant API ({}) to confirm model status.".format(
                    "https://www.pinecone.io/docs/api/operation/assistant/describe_assistant/"
                )
            )

        return AssistantModel(
            assistant=assistant,
            client_builder=self._client_builder,
            config=self.config,
        )

    def describe_assistant(self, assistant_name: str) -> AssistantModel:
        """Describe an existing assistant by name."""
        assistant = self._assistant_control_api.get_assistant(assistant_name=assistant_name)
        return AssistantModel(
            assistant=assistant,
            client_builder=self._client_builder,
            config=self.config,
        )

    def list_assistants(self) -> List[AssistantModel]:
        """List all assistants (auto-paginates)."""
        all_assistants: list[Any] = []
        pagination_token = None
        while True:
            page = self.list_assistants_paginated(pagination_token=pagination_token)
            all_assistants.extend(page.assistants)
            if page.next_token is None:
                break
            pagination_token = page.next_token
        return all_assistants

    def list_assistants_paginated(
        self,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
    ) -> ListAssistantsResponse:
        """List one page of assistants."""
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        if pagination_token is not None:
            kwargs["pagination_token"] = pagination_token

        resp = self._assistant_control_api.list_assistants(**kwargs)
        return ListAssistantsResponse.from_openapi(resp, self._client_builder, self.config)

    def update_assistant(
        self,
        assistant_name: str,
        instructions: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AssistantModel:
        """Update an existing assistant's instructions or metadata."""
        request_body = dict(instructions=instructions, metadata=metadata)  # noqa: C408
        assistant = self._assistant_control_api.update_assistant(
            assistant_name=assistant_name, update_assistant_request=request_body
        )
        return AssistantModel(
            assistant=assistant,
            client_builder=self._client_builder,
            config=self.config,
        )

    def delete_assistant(
        self,
        assistant_name: str,
        timeout: Optional[int] = None,
    ) -> None:
        """Delete an assistant and optionally wait until it is gone."""
        self._assistant_control_api.delete_assistant(assistant_name=assistant_name)

        if timeout == -1:
            return

        if timeout is None:
            assistant = self.describe_assistant(assistant_name)
            while assistant:
                time.sleep(5)
                try:
                    assistant = self.describe_assistant(assistant_name)
                except Exception:
                    assistant = None  # type: ignore[assignment]
        else:
            assistant = self.describe_assistant(assistant_name)
            while assistant and timeout >= 0:
                time.sleep(5)
                timeout -= 5
                try:
                    assistant = self.describe_assistant(assistant_name)
                except Exception:
                    assistant = None  # type: ignore[assignment]

        if timeout and timeout < 0:
            raise TimeoutError(
                "Please call the describe_assistant API ({}) to confirm model status.".format(
                    "https://www.pinecone.io/docs/api/operation/assistant/describe_assistant/"
                )
            )

    def Assistant(self, assistant_name: str) -> AssistantModel:  # noqa: N802
        """Alias for ``describe_assistant``."""
        return self.describe_assistant(assistant_name)
