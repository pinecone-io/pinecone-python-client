"""Model information response models for the Inference API."""

from __future__ import annotations

from typing import Any, cast

import msgspec
from msgspec import Struct


class ModelInfoSupportedParameter(Struct, kw_only=True):
    """A supported parameter for an inference model.

    Attributes:
        parameter: The parameter name.
        type: The parameter category (e.g. ``"one_of"``).
        value_type: The value type (e.g. ``"string"``).
        required: Whether the parameter is required.
        allowed_values: Allowed values for enum-style parameters.
        min: Minimum value for numeric parameters.
        max: Maximum value for numeric parameters.
        default: Default value for the parameter.
    """

    parameter: str
    type: str
    value_type: str
    required: bool
    allowed_values: list[str | int] | None = None
    min: float | None = None
    max: float | None = None
    default: str | int | float | bool | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. param['parameter'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'parameter' in param``)."""
        return key in self.__struct_fields__


_MODEL_INFO_ALIASES: dict[str, str] = {"name": "model", "description": "short_description"}


class ModelInfo(Struct, kw_only=True):
    """Information about an inference model.

    Attributes:
        model: The model identifier (also accessible as ``name``).
        short_description: A brief description of the model (also accessible as ``description``).
        type: The model type (e.g. ``"embed"``, ``"rerank"``).
        supported_parameters: Parameters accepted by the model.
        vector_type: The type of vectors produced (for embed models).
        default_dimension: Default output dimension (for embed models).
        supported_dimensions: Available output dimensions (for embed models).
        modality: The input modality (e.g. ``"text"``).
        max_sequence_length: Maximum input sequence length.
        max_batch_size: Maximum batch size for requests.
        provider_name: The model provider.
        supported_metrics: Supported similarity metrics.
    """

    model: str
    short_description: str
    type: str
    supported_parameters: list[ModelInfoSupportedParameter]
    vector_type: str | None = None
    default_dimension: int | None = None
    supported_dimensions: list[int] | None = None
    modality: str | None = None
    max_sequence_length: int | None = None
    max_batch_size: int | None = None
    provider_name: str | None = None
    supported_metrics: list[str] | None = None

    @property
    def name(self) -> str:
        """Alias for ``model`` — the model identifier."""
        return self.model

    @property
    def description(self) -> str:
        """Alias for ``short_description`` — a brief description of the model."""
        return self.short_description

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. model_info['model'])."""
        key = _MODEL_INFO_ALIASES.get(key, key)
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'model' in model_info``)."""
        if isinstance(key, str):
            key = _MODEL_INFO_ALIASES.get(key, key)
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this object."""
        return cast(dict[str, Any], msgspec.to_builtins(self))

    def __getattr__(self, name: str) -> Any:
        """Legacy alias passthrough and AttributeError for unknown attributes."""
        resolved = _MODEL_INFO_ALIASES.get(name)
        if resolved is not None:
            return getattr(self, resolved)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
