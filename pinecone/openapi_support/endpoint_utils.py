from .model_utils import file_type
from .exceptions import PineconeApiTypeError, PineconeApiValueError
from typing import Optional, Dict, Tuple, TypedDict, List, Literal
from .types import PropertyValidationTypedDict
from .model_utils import validate_and_convert_types, check_allowed_values, check_validations


class ExtraOpenApiKwargsTypedDict(TypedDict, total=False):
    _return_http_data_only: Optional[bool]
    _preload_content: Optional[bool]
    _request_timeout: Optional[int]
    _check_input_type: Optional[bool]
    _check_return_type: Optional[bool]
    _host_index: Optional[int]
    async_req: Optional[bool]


class KwargsWithOpenApiKwargDefaultsTypedDict(TypedDict, total=False):
    _return_http_data_only: bool
    _preload_content: bool
    _request_timeout: int
    _check_input_type: bool
    _check_return_type: bool
    _host_index: Optional[int]
    async_req: bool


class EndpointSettingsDict(TypedDict):
    response_type: Optional[Tuple]
    auth: List[str]
    endpoint_path: str
    operation_id: str
    http_method: Literal["POST", "PUT", "PATCH", "GET", "DELETE"]
    servers: Optional[List[str]]


class EndpointParamsMapDict(TypedDict):
    all: List[str]
    required: List[str]
    nullable: List[str]
    enum: List[str]
    validation: List[str]


class EndpointRootMapDict(TypedDict):
    validations: Dict[Tuple[str], PropertyValidationTypedDict]
    allowed_values: Dict[Tuple[str], Dict]
    openapi_types: Dict[str, Tuple]
    attribute_map: Dict[str, str]
    location_map: Dict[str, str]
    collection_format_map: Dict[str, str]


class EndpointUtils:
    @staticmethod
    def gather_params(attribute_map, location_map, openapi_types, collection_format_map, kwargs):
        params = {
            "body": None,
            "collection_format": {},
            "file": {},
            "form": [],
            "header": {},
            "path": {},
            "query": [],
        }

        for param_name, param_value in kwargs.items():
            param_location = location_map.get(param_name)
            if param_location is None:
                continue
            if param_location:
                if param_location == "body":
                    params["body"] = param_value
                    continue
                base_name = attribute_map[param_name]
                if param_location == "form" and openapi_types[param_name] == (file_type,):
                    params["file"][param_name] = [param_value]
                elif param_location == "form" and openapi_types[param_name] == ([file_type],):
                    # param_value is already a list
                    params["file"][param_name] = param_value
                elif param_location in {"form", "query"}:
                    param_value_full = (base_name, param_value)
                    params[param_location].append(param_value_full)
                if param_location not in {"form", "query"}:
                    params[param_location][base_name] = param_value
                collection_format = collection_format_map.get(param_name)
                if collection_format:
                    params["collection_format"][base_name] = collection_format

        return params

    @staticmethod
    def raise_if_missing_required_params(params_map, settings, kwargs):
        for key in params_map["required"]:
            if key not in kwargs.keys():
                raise PineconeApiValueError(
                    "Missing the required parameter `%s` when calling "
                    "`%s`" % (key, settings["operation_id"])
                )

    @staticmethod
    def raise_if_unexpected_param(params_map, settings, kwargs):
        for key, value in kwargs.items():
            if key not in params_map["all"]:
                raise PineconeApiTypeError(
                    "Got an unexpected parameter '%s'"
                    " to method `%s`" % (key, settings["operation_id"])
                )
            # only throw this nullable PineconeApiValueError if _check_input_type
            # is False, if _check_input_type==True we catch this case
            # in self.__validate_inputs
            if (
                key not in params_map["nullable"]
                and value is None
                and kwargs["_check_input_type"] is False
            ):
                raise PineconeApiValueError(
                    "Value may not be None for non-nullable parameter `%s`"
                    " when calling `%s`" % (key, settings["operation_id"])
                )

    @staticmethod
    def raise_if_invalid_inputs(
        config, params_map, allowed_values, validations, openapi_types, kwargs
    ):
        for param in params_map["enum"]:
            if param in kwargs:
                check_allowed_values(allowed_values, (param,), kwargs[param])

        for param in params_map["validation"]:
            if param in kwargs:
                check_validations(validations, (param,), kwargs[param], configuration=config)

        if kwargs["_check_input_type"] is False:
            return

        for key, value in kwargs.items():
            fixed_val = validate_and_convert_types(
                value,
                openapi_types[key],
                [key],
                False,
                kwargs["_check_input_type"],
                configuration=config,
            )
            kwargs[key] = fixed_val
