# coding: utf-8

"""
    Pinecone API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

    The version of the OpenAPI document: version not set
    Contact: support@pinecone.io
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, StrictInt, StrictStr
from pinecone.core.client.models.index_meta_database_index_config import IndexMetaDatabaseIndexConfig


class IndexMetaDatabase(BaseModel):
    """
    IndexMetaDatabase
    """

    name: StrictStr = Field(...)
    dimension: StrictStr = Field(...)
    capacity_mode: StrictStr = Field(...)
    index_type: Optional[StrictStr] = None
    metric: StrictStr = Field(...)
    pods: Optional[StrictInt] = None
    replicas: Optional[StrictInt] = None
    shards: Optional[StrictInt] = None
    pod_type: Optional[StrictStr] = None
    index_config: Optional[IndexMetaDatabaseIndexConfig] = None
    metadata_config: Optional[Dict[str, Any]] = None
    __properties = [
        "name",
        "dimension",
        "capacity_mode",
        "index_type",
        "metric",
        "pods",
        "replicas",
        "shards",
        "pod_type",
        "index_config",
        "metadata_config",
    ]

    class Config:
        """Pydantic configuration"""

        allow_population_by_field_name = True
        validate_assignment = True

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> IndexMetaDatabase:
        """Create an instance of IndexMetaDatabase from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True, exclude={}, exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of index_config
        if self.index_config:
            _dict["index_config"] = self.index_config.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> IndexMetaDatabase:
        """Create an instance of IndexMetaDatabase from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return IndexMetaDatabase.parse_obj(obj)

        _obj = IndexMetaDatabase.parse_obj(
            {
                "name": obj.get("name"),
                "dimension": obj.get("dimension"),
                "capacity_mode": obj.get("capacity_mode"),
                "index_type": obj.get("index_type"),
                "metric": obj.get("metric") if obj.get("metric") is not None else "cosine",
                "pods": obj.get("pods"),
                "replicas": obj.get("replicas"),
                "shards": obj.get("shards"),
                "pod_type": obj.get("pod_type"),
                "index_config": IndexMetaDatabaseIndexConfig.from_dict(obj.get("index_config"))
                if obj.get("index_config") is not None
                else None,
                "metadata_config": obj.get("metadata_config"),
            }
        )
        return _obj
