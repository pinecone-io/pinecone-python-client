"""Unit tests for OpenAPI endpoint validation logic.

These tests replace integration tests that were making real API calls to test
client-side validation. They test the validation logic directly without
requiring API access.
"""

import pytest
from pinecone.openapi_support.endpoint_utils import (
    EndpointUtils,
    EndpointParamsMapDict,
    EndpointSettingsDict,
    AllowedValuesDict,
    OpenapiTypesDictType,
)
from pinecone.openapi_support.types import PropertyValidationTypedDict
from pinecone.config.openapi_configuration import Configuration
from pinecone.exceptions import PineconeApiTypeError, PineconeApiValueError


class TestEndpointUtilsTypeValidation:
    """Test type validation in EndpointUtils.raise_if_invalid_inputs"""

    def test_raise_if_invalid_inputs_with_wrong_type(self):
        """Test that PineconeApiTypeError is raised when wrong type is passed"""
        config = Configuration()
        params_map: EndpointParamsMapDict = {
            "all": ["dimension", "_check_input_type"],
            "required": [],
            "nullable": [],
            "enum": [],
            "validation": [],
        }
        allowed_values: AllowedValuesDict = {}
        validations: PropertyValidationTypedDict = {}
        openapi_types: OpenapiTypesDictType = {"dimension": (int,), "_check_input_type": (bool,)}
        kwargs = {
            "dimension": "10",  # String instead of int
            "_check_input_type": True,
        }

        with pytest.raises(PineconeApiTypeError) as exc_info:
            EndpointUtils.raise_if_invalid_inputs(
                config, params_map, allowed_values, validations, openapi_types, kwargs
            )

        assert "dimension" in str(exc_info.value).lower() or "Invalid type" in str(exc_info.value)

    def test_raise_if_invalid_inputs_with_correct_type(self):
        """Test that no error is raised when correct type is passed"""
        config = Configuration()
        params_map: EndpointParamsMapDict = {
            "all": ["dimension", "_check_input_type"],
            "required": [],
            "nullable": [],
            "enum": [],
            "validation": [],
        }
        allowed_values: AllowedValuesDict = {}
        validations: PropertyValidationTypedDict = {}
        openapi_types: OpenapiTypesDictType = {"dimension": (int,), "_check_input_type": (bool,)}
        kwargs = {
            "dimension": 10,  # Correct type
            "_check_input_type": True,
        }

        # Should not raise
        EndpointUtils.raise_if_invalid_inputs(
            config, params_map, allowed_values, validations, openapi_types, kwargs
        )

    def test_raise_if_invalid_inputs_with_type_check_disabled(self):
        """Test that type checking can be disabled"""
        config = Configuration()
        params_map: EndpointParamsMapDict = {
            "all": ["dimension", "_check_input_type"],
            "required": [],
            "nullable": [],
            "enum": [],
            "validation": [],
        }
        allowed_values: AllowedValuesDict = {}
        validations: PropertyValidationTypedDict = {}
        openapi_types: OpenapiTypesDictType = {"dimension": (int,), "_check_input_type": (bool,)}
        kwargs = {
            "dimension": "10",  # Wrong type but checking disabled
            "_check_input_type": False,
        }

        # Should not raise when _check_input_type is False
        EndpointUtils.raise_if_invalid_inputs(
            config, params_map, allowed_values, validations, openapi_types, kwargs
        )

    def test_raise_if_missing_required_params(self):
        """Test that PineconeApiValueError is raised when required param is missing"""
        params_map: EndpointParamsMapDict = {
            "all": ["dimension", "name"],
            "required": ["dimension", "name"],
            "nullable": [],
            "enum": [],
            "validation": [],
        }
        settings: EndpointSettingsDict = {
            "response_type": None,
            "auth": [],
            "endpoint_path": "/indexes",
            "operation_id": "create_index",
            "http_method": "POST",
            "servers": None,
        }
        kwargs = {
            "name": "test-index"
            # dimension is missing
        }

        with pytest.raises(PineconeApiValueError) as exc_info:
            EndpointUtils.raise_if_missing_required_params(params_map, settings, kwargs)

        assert "dimension" in str(exc_info.value)
        assert "create_index" in str(exc_info.value)

    def test_raise_if_unexpected_param(self):
        """Test that PineconeApiTypeError is raised for unexpected parameters"""
        params_map: EndpointParamsMapDict = {
            "all": ["dimension", "name"],
            "required": [],
            "nullable": [],
            "enum": [],
            "validation": [],
        }
        settings: EndpointSettingsDict = {
            "response_type": None,
            "auth": [],
            "endpoint_path": "/indexes",
            "operation_id": "create_index",
            "http_method": "POST",
            "servers": None,
        }
        kwargs = {
            "dimension": 10,
            "name": "test-index",
            "unexpected_param": "value",  # Not in params_map["all"]
            "_check_input_type": True,
        }

        with pytest.raises(PineconeApiTypeError) as exc_info:
            EndpointUtils.raise_if_unexpected_param(params_map, settings, kwargs)

        assert "unexpected_param" in str(exc_info.value)
        assert "create_index" in str(exc_info.value)

    def test_raise_if_invalid_inputs_with_enum_validation(self):
        """Test enum value validation"""
        config = Configuration()
        params_map: EndpointParamsMapDict = {
            "all": ["metric", "_check_input_type"],
            "required": [],
            "nullable": [],
            "enum": ["metric"],
            "validation": [],
        }
        allowed_values: AllowedValuesDict = {
            ("metric",): {"cosine": "cosine", "euclidean": "euclidean", "dotproduct": "dotproduct"}
        }
        validations: PropertyValidationTypedDict = {}
        openapi_types: OpenapiTypesDictType = {"metric": (str,), "_check_input_type": (bool,)}
        kwargs = {
            "metric": "invalid_metric",  # Not in allowed values
            "_check_input_type": True,
        }

        with pytest.raises(PineconeApiValueError) as exc_info:
            EndpointUtils.raise_if_invalid_inputs(
                config, params_map, allowed_values, validations, openapi_types, kwargs
            )

        assert "metric" in str(exc_info.value).lower()
        assert "invalid" in str(exc_info.value).lower()

    def test_raise_if_invalid_inputs_with_enum_valid_value(self):
        """Test that valid enum values pass validation"""
        config = Configuration()
        params_map: EndpointParamsMapDict = {
            "all": ["metric", "_check_input_type"],
            "required": [],
            "nullable": [],
            "enum": ["metric"],
            "validation": [],
        }
        allowed_values: AllowedValuesDict = {
            ("metric",): {"cosine": "cosine", "euclidean": "euclidean", "dotproduct": "dotproduct"}
        }
        validations: PropertyValidationTypedDict = {}
        openapi_types: OpenapiTypesDictType = {"metric": (str,), "_check_input_type": (bool,)}
        kwargs = {
            "metric": "cosine",  # Valid enum value
            "_check_input_type": True,
        }

        # Should not raise
        EndpointUtils.raise_if_invalid_inputs(
            config, params_map, allowed_values, validations, openapi_types, kwargs
        )
