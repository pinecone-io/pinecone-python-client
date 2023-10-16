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


from typing import Dict, Optional, Union
from pydantic import BaseModel, Field, StrictFloat, StrictInt
from pinecone.core.client.models.namespace_summary import NamespaceSummary


class DescribeIndexStatsResponse(BaseModel):
    """
    The response for the `DescribeIndexStats` operation.  # noqa: E501
    """

    namespaces: Optional[Dict[str, NamespaceSummary]] = Field(
        None,
        description="A mapping for each namespace in the index from the namespace name to a summary of its contents. If a metadata filter expression is present, the summary will reflect only vectors matching that expression.",
    )
    dimension: Optional[StrictInt] = Field(None, description="The dimension of the indexed vectors.")
    index_fullness: Optional[Union[StrictFloat, StrictInt]] = Field(
        None,
        alias="indexFullness",
        description="The fullness of the index, regardless of whether a metadata filter expression was passed. The granularity of this metric is 10%.",
    )
    total_vector_count: Optional[StrictInt] = Field(None, alias="totalVectorCount")
    __properties = ["namespaces", "dimension", "indexFullness", "totalVectorCount"]

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
    def from_json(cls, json_str: str) -> DescribeIndexStatsResponse:
        """Create an instance of DescribeIndexStatsResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True, exclude={}, exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each value in namespaces (dict)
        _field_dict = {}
        if self.namespaces:
            for _key in self.namespaces:
                if self.namespaces[_key]:
                    _field_dict[_key] = self.namespaces[_key].to_dict()
            _dict["namespaces"] = _field_dict
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DescribeIndexStatsResponse:
        """Create an instance of DescribeIndexStatsResponse from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DescribeIndexStatsResponse.parse_obj(obj)

        _obj = DescribeIndexStatsResponse.parse_obj(
            {
                "namespaces": dict((_k, NamespaceSummary.from_dict(_v)) for _k, _v in obj.get("namespaces").items())
                if obj.get("namespaces") is not None
                else None,
                "dimension": obj.get("dimension"),
                "index_fullness": obj.get("indexFullness"),
                "total_vector_count": obj.get("totalVectorCount"),
            }
        )
        return _obj
