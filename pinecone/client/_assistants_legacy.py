"""Backwards-compatibility method shims for the Assistants namespace.

Each method here mirrors a legacy method name from the removed
pinecone_plugins.assistant package. They delegate to the canonical
new-SDK method and remap legacy parameter names.

Keep this mixin narrowly scoped — it should contain only aliases, never
new behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pinecone.models.assistant.model import AssistantModel


class AssistantsLegacyNamespaceMixin:
    """Legacy-name method shims for the :class:`Assistants` namespace.

    Mixed into :class:`Assistants` so that callers upgrading from
    ``pinecone_plugins.assistant`` can keep using names like
    ``create_assistant`` and parameter names like ``assistant_name``.
    """

    def create_assistant(
        self,
        assistant_name: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        region: str = "us",
        timeout: int | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Deprecated alias for :meth:`Assistants.create`.

        Accepts either ``assistant_name`` (legacy) or ``name`` (current),
        but not both. All other parameters are forwarded unchanged.
        """
        resolved_name = assistant_name if assistant_name is not None else name
        return cast(
            "AssistantModel",
            self.create(  # type: ignore[attr-defined]
                name=resolved_name,
                instructions=instructions,
                metadata=metadata,
                region=region,
                timeout=timeout,
                **kwargs,
            ),
        )
