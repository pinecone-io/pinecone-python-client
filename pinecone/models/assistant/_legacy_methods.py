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
    from pinecone.models.assistant.file_model import AssistantFileModel


class AssistantModelLegacyMethodsMixin:
    """Legacy method aliases for :class:`AssistantModel`.

    Individual methods are added by BC-0016..BC-0023. This base scaffolds
    the ``_assistants`` back-reference and a helper to resolve it.
    """

    # Declared ClassVar so msgspec ignores it when reading __struct_fields__.
    _assistants_ref: ClassVar[Any | None] = None

    def _resolve_assistants(self) -> Assistants:
        """Return the owning :class:`Assistants` namespace.

        Raises:
            RuntimeError: If the model was constructed without a back-reference
                (e.g. decoded directly from JSON in user code), making legacy
                delegation impossible.
        """
        ref: Assistants | None = getattr(self, "_assistants", None)
        if ref is None:
            raise RuntimeError(
                "This AssistantModel has no client reference, so legacy "
                "methods cannot delegate. Use pc.assistants.<method>(...) "
                "directly, or obtain the model via pc.assistants.describe(name=...)."
            )
        return ref

    def describe_file(
        self,
        file_id: str,
        include_url: bool = False,
        **kwargs: Any,
    ) -> "AssistantFileModel":
        """Deprecated alias for :meth:`Assistants.describe_file`."""
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
        """Deprecated alias — upload a byte stream as a file.

        In the new SDK, byte streams are uploaded via the unified
        :meth:`Assistants.upload_file` using ``file_stream=`` and ``file_name=``.
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

    def upload_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        timeout: int | None = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Deprecated alias — upload a file to this assistant.

        Equivalent to ``pc.assistants.upload_file(assistant_name=self.name, ...)``.
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
