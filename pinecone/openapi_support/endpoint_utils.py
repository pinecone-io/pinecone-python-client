from .model_utils import file_type
from .exceptions import PineconeApiTypeError, PineconeApiValueError
from typing import TypedDict, Literal, Any
from .types import PropertyValidationTypedDict
from ..config.openapi_configuration import Configuration
from .model_utils import validate_and_convert_types, check_allowed_values, check_validations


class ExtraOpenApiKwargsTypedDict(TypedDict, total=False):
    _return_http_data_only: bool | None
    _preload_content: bool | None
    _request_timeout: int | None
    _check_input_type: bool | None
    _check_return_type: bool | None
    async_req: bool | None


class KwargsWithOpenApiKwargDefaultsTypedDict(TypedDict, total=False):
    _return_http_data_only: bool
    _preload_content: bool
    _request_timeout: int
    _check_input_type: bool
    _check_return_type: bool
    async_req: bool


class EndpointSettingsDict(TypedDict):
    response_type: tuple | None
    auth: list[str]
    endpoint_path: str
    operation_id: str
    http_method: Literal["POST", "PUT", "PATCH", "GET", "DELETE"]
    servers: list[str] | None


class EndpointParamsMapDict(TypedDict):
    all: list[str]
    required: list[str]
    nullable: list[str]
    enum: list[str]
    validation: list[str]


AllowedValuesDict = dict[tuple[str], dict]

AttributeMapDictType = dict[str, str]
LocationMapDictType = dict[str, str]
OpenapiTypesDictType = dict[str, tuple]


class EndpointRootMapDict(TypedDict):
    validations: dict[tuple[str], PropertyValidationTypedDict]
    allowed_values: dict[tuple[str], dict]
    openapi_types: OpenapiTypesDictType
    attribute_map: AttributeMapDictType
    location_map: LocationMapDictType
    collection_format_map: dict[str, str]


class CombinedParamsMapDict(TypedDict):
    body: Any
    collection_format: dict[str, str]
    file: dict[str, list[file_type]]
    form: list[tuple[str, Any]]
    header: dict[str, list[str]]
    path: dict[str, Any]
    query: list[tuple[str, Any]]


class EndpointUtils:
    @staticmethod
    def gather_params(
        attribute_map: AttributeMapDictType,
        location_map: LocationMapDictType,
        openapi_types: OpenapiTypesDictType,
        collection_format_map: dict[str, str],
        kwargs: dict[str, Any],
    ) -> CombinedParamsMapDict:
        params: CombinedParamsMapDict = {
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
                elif param_location == "form":
                    param_value_full = (base_name, param_value)
                    params["form"].append(param_value_full)
                elif param_location == "query":
                    param_value_full = (base_name, param_value)
                    params["query"].append(param_value_full)
                elif param_location == "header":
                    params["header"][base_name] = param_value
                elif param_location == "path":
                    params["path"][base_name] = param_value
                else:
                    raise PineconeApiTypeError("Got an unexpected location '%s' for parameter `%s`")

                collection_format = collection_format_map.get(param_name)
                if collection_format:
                    params["collection_format"][base_name] = collection_format

        return params

    @staticmethod
    def raise_if_missing_required_params(
        params_map: EndpointParamsMapDict, settings: EndpointSettingsDict, kwargs: dict[str, Any]
    ) -> None:
        for key in params_map["required"]:
            if key not in kwargs.keys():
                raise PineconeApiValueError(
                    "Missing the required parameter `%s` when calling "
                    "`%s`" % (key, settings["operation_id"])
                )

    @staticmethod
    def raise_if_unexpected_param(
        params_map: EndpointParamsMapDict, settings: EndpointSettingsDict, kwargs: dict[str, Any]
    ) -> None:
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
        config: Configuration,
        params_map: EndpointParamsMapDict,
        allowed_values: AllowedValuesDict,
        validations: dict[tuple[str], PropertyValidationTypedDict],
        openapi_types: OpenapiTypesDictType,
        kwargs: dict[str, Any],
    ) -> None:
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
