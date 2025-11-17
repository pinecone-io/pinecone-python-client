from dataclasses import dataclass
from typing import Any

from pinecone.db_control.enums import Metric
from pinecone.inference.inference_request_builder import EmbedModel


@dataclass(frozen=True)
class IndexEmbed:
    """
    IndexEmbed represents the index embedding configuration when creating an index from a model.
    """

    model: str
    """
    The name of the embedding model to use for the index.
    Required.
    """

    field_map: dict[str, Any]
    """
    A mapping of field names to their types.
    Required.
    """

    metric: str | None = None
    """
    The metric to use for the index. If not provided, the default metric for the model is used.
    Optional.
    """

    read_parameters: dict[str, Any] | None = None
    """
    The parameters to use when reading from the index.
    Optional.
    """

    write_parameters: dict[str, Any] | None = None
    """
    The parameters to use when writing to the index.
    Optional.
    """

    def as_dict(self) -> dict[str, Any]:
        """
        Returns the IndexEmbed as a dictionary.
        """
        return self.__dict__

    def __init__(
        self,
        model: EmbedModel | str,
        field_map: dict[str, Any],
        metric: (Metric | str) | None = None,
        read_parameters: dict[str, Any] | None = None,
        write_parameters: dict[str, Any] | None = None,
    ):
        object.__setattr__(
            self, "model", model.value if isinstance(model, EmbedModel) else str(model)
        )
        object.__setattr__(self, "field_map", field_map)
        object.__setattr__(self, "metric", metric.value if isinstance(metric, Metric) else metric)
        object.__setattr__(
            self, "read_parameters", read_parameters if read_parameters is not None else {}
        )
        object.__setattr__(
            self, "write_parameters", write_parameters if write_parameters is not None else {}
        )
