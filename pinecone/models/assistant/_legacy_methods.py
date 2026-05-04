"""Backwards-compatibility method shims for :class:`AssistantModel`.

Legacy callers invoked data-plane operations directly on the assistant
object (``assistant.upload_file(...)``, ``assistant.chat(...)``).
In the new SDK these live on the :class:`Assistants` namespace and
take the assistant name as a parameter. Each method in this mixin
delegates to the namespace using ``self.name``.

Back-reference storage:
    msgspec Struct instances do not have a ``__dict__`` by default and
    their ``__setattr__`` only allows setting declared struct fields.
    ``AssistantModel`` is declared with ``dict=True`` which adds a
    ``__dict__`` to each instance. :meth:`Assistants._attach_ref` writes
    directly into ``model.__dict__["_assistants"]`` to store the reference
    without going through msgspec's ``__setattr__``.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pinecone.client.assistants import Assistants
    from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
    from pinecone.models.assistant.context import ContextResponse
    from pinecone.models.assistant.file_model import AssistantFileModel
    from pinecone.models.assistant.list import ListFilesResponse
    from pinecone.models.assistant.message import Message
    from pinecone.models.assistant.options import ContextOptions
    from pinecone.models.assistant.streaming import (
        ChatCompletionStream,
        ChatStream,
    )


class AssistantModelLegacyMethodsMixin:
    """Legacy method aliases for :class:`AssistantModel`.

    Individual methods are added by BC-0016..BC-0023. This base scaffolds
    the ``_assistants`` back-reference and a helper to resolve it.
    """

    # Declared ClassVar so msgspec ignores it when reading __struct_fields__.
    _assistants_ref: ClassVar[Any | None] = None

    def _resolve_assistants(self) -> Assistants:
        """Return the owning sync :class:`Assistants` namespace.

        Raises:
            RuntimeError: If the model has no client reference at all.
            TypeError: If the back-reference is an :class:`AsyncAssistants`
                instance — legacy shims are sync-only; async callers must use
                the namespace method directly (e.g. ``await pc.assistants.chat(
                assistant_name=model.name, ...)``).
        """

        # AsyncAssistants is imported lazily to avoid a circular import at module level.
        ref: Assistants | None = getattr(self, "_assistants", None)
        if ref is None:
            raise RuntimeError(
                "This AssistantModel has no client reference, so legacy "
                "methods cannot delegate. Use pc.assistants.<method>(...) "
                "directly, or obtain the model via "
                "pc.assistants.describe(name=...)."
            )
        from pinecone.async_client.assistants import AsyncAssistants

        if isinstance(ref, AsyncAssistants):
            raise TypeError(
                "Legacy assistant methods on AssistantModel are sync-only "
                "and cannot be used on a model retrieved from AsyncAssistants. "
                "Use the async namespace directly: "
                "await pc.assistants.<method>(assistant_name=model.name, ...)."
            )
        return ref

    def describe_file(
        self,
        file_id: str,
        include_url: bool = False,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Describe a file associated with this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.describe_file` instead.
        """
        ns = self._resolve_assistants()
        return ns.describe_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_id=file_id,
            include_url=include_url,
            **kwargs,
        )

    def upload_bytes_stream(
        self,
        stream: IO[bytes],
        file_name: str,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        timeout: int | None = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Upload a byte stream as a file to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.upload_file` with ``file_stream=`` and
            ``file_name=`` instead.
        """
        ns = self._resolve_assistants()
        return ns.upload_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_stream=stream,
            file_name=file_name,
            metadata=metadata,
            multimodal=multimodal,
            timeout=timeout,
            file_id=file_id,
            **kwargs,
        )

    def list_files(
        self,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[AssistantFileModel]:
        """Return a materialized list of files for this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.list_files` instead. Note that the namespace
            method returns a lazy paginator; iterate over it to get all files.
        """
        ns = self._resolve_assistants()
        return list(
            ns.list_files(
                assistant_name=self.name,  # type: ignore[attr-defined]
                filter=filter,
            )
        )

    def list_files_paginated(
        self,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        *,
        page_size: int | None = None,
        **kwargs: Any,
    ) -> ListFilesResponse:
        """Return a single page of files for this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.list_files_page` instead. The ``limit`` and
            ``page_size`` parameters are accepted for backwards compatibility but
            are not forwarded; the underlying endpoint does not support page-size
            control.
        """
        ns = self._resolve_assistants()
        return ns.list_files_page(
            assistant_name=self.name,  # type: ignore[attr-defined]
            filter=filter,
            pagination_token=pagination_token,
        )

    def upload_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        timeout: int | None = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Upload a file to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.upload_file` instead.
        """
        ns = self._resolve_assistants()
        return ns.upload_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_path=file_path,
            metadata=metadata,
            multimodal=multimodal,
            timeout=timeout,
            file_id=file_id,
            **kwargs,
        )

    def delete_file(
        self,
        file_id: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete a file from this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.delete_file` instead.
        """
        ns = self._resolve_assistants()
        ns.delete_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_id=file_id,
            timeout=timeout,
            **kwargs,
        )

    def chat_completions(
        self,
        messages: list[Message] | list[dict[str, Any]],
        filter: dict[str, Any] | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse | ChatCompletionStream:
        """Send a chat-completions request to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.chat_completions` instead.
        """
        ns = self._resolve_assistants()
        return ns.chat_completions(
            assistant_name=self.name,  # type: ignore[attr-defined]
            messages=messages,
            filter=filter,
            stream=stream,
            model=model,  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )

    def context(
        self,
        query: str,
        filter: dict[str, Any] | None = None,
        top_k: int | None = None,
        snippet_size: int | None = None,
        context_options: ContextOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ContextResponse:
        """Retrieve context snippets from this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.context` instead. The ``context_options``
            parameter is unpacked into ``multimodal`` and
            ``include_binary_content`` for the new API.
        """
        ns = self._resolve_assistants()
        # Unpack context_options into the new API's individual parameters.
        multimodal: bool | None = None
        include_binary_content: bool | None = None
        if context_options is not None:
            if isinstance(context_options, dict):
                multimodal = context_options.get("multimodal")
                include_binary_content = context_options.get("include_binary_content")
                if top_k is None:
                    top_k = context_options.get("top_k")
                if snippet_size is None:
                    snippet_size = context_options.get("snippet_size")
            else:
                multimodal = context_options.multimodal
                include_binary_content = context_options.include_binary_content
                if top_k is None:
                    top_k = context_options.top_k
                if snippet_size is None:
                    snippet_size = context_options.snippet_size
        return ns.context(
            assistant_name=self.name,  # type: ignore[attr-defined]
            query=query,
            filter=filter,
            top_k=top_k,
            snippet_size=snippet_size,
            multimodal=multimodal,
            include_binary_content=include_binary_content,
        )

    def chat(
        self,
        messages: list[Message] | list[dict[str, Any]],
        filter: dict[str, Any] | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: ContextOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | ChatStream:
        """Send a chat request to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.chat` instead.
        """
        ns = self._resolve_assistants()
        return ns.chat(
            assistant_name=self.name,  # type: ignore[attr-defined]
            messages=messages,
            filter=filter,
            stream=stream,
            model=model,  # type: ignore[arg-type]
            temperature=temperature,
            json_response=json_response,
            include_highlights=include_highlights,
            context_options=context_options,
            **kwargs,
        )
