import json
from pinecone.utils.repr_overrides import custom_serializer, install_json_repr_override
from pinecone.core.openapi.inference.model.model_info import ModelInfo as OpenAPIModelInfo
from pinecone.core.openapi.inference.model.model_info_supported_parameter import (
    ModelInfoSupportedParameter as OpenAPIModelInfoSupportedParameter,
)

for klass in [
    # OpenAPIModelInfo,
    # OpenAPIModelInfoMetric,
    OpenAPIModelInfoSupportedParameter
    # OpenAPIModelInfoSupportedMetrics,
]:
    install_json_repr_override(klass)


class ModelInfo:
    def __init__(self, model_info: OpenAPIModelInfo):
        self._model_info = model_info
        self.supported_metrics: list[str] = []
        if self._model_info.supported_metrics is not None:
            # Handle both cases: list of strings (Python 3.13+) or list of enum-like objects
            metrics_value = self._model_info.supported_metrics.value
            if metrics_value is not None:
                for sm in metrics_value:
                    if isinstance(sm, str):
                        self.supported_metrics.append(sm)
                    elif hasattr(sm, "value"):
                        self.supported_metrics.append(sm.value)
                    else:
                        # Fallback: use the value as-is
                        self.supported_metrics.append(sm)

    def __str__(self):
        return str(self._model_info)

    def __getattr__(self, attr):
        if attr == "supported_metrics":
            return self.supported_metrics
        else:
            return getattr(self._model_info, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        raw = self._model_info.to_dict()
        raw["supported_metrics"] = self.supported_metrics
        return raw
