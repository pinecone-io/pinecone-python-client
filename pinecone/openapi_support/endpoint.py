from .exceptions import PineconeApiValueError, PineconeApiTypeError
from .model_utils import (
    none_type,
    file_type,
    check_allowed_values,
    validate_and_convert_types,
    check_validations,
)


class Endpoint(object):
    def __init__(
        self,
        settings=None,
        params_map=None,
        root_map=None,
        headers_map=None,
        api_client=None,
        callable=None,
    ):
        """Creates an endpoint

        Args:
            settings (dict): see below key value pairs
                'response_type' (tuple/None): response type
                'auth' (list): a list of auth type keys
                'endpoint_path' (str): the endpoint path
                'operation_id' (str): endpoint string identifier
                'http_method' (str): POST/PUT/PATCH/GET etc
                'servers' (list): list of str servers that this endpoint is at
            params_map (dict): see below key value pairs
                'all' (list): list of str endpoint parameter names
                'required' (list): list of required parameter names
                'nullable' (list): list of nullable parameter names
                'enum' (list): list of parameters with enum values
                'validation' (list): list of parameters with validations
            root_map
                'validations' (dict): the dict mapping endpoint parameter tuple
                    paths to their validation dictionaries
                'allowed_values' (dict): the dict mapping endpoint parameter
                    tuple paths to their allowed_values (enum) dictionaries
                'openapi_types' (dict): param_name to openapi type
                'attribute_map' (dict): param_name to camelCase name
                'location_map' (dict): param_name to  'body', 'file', 'form',
                    'header', 'path', 'query'
                collection_format_map (dict): param_name to `csv` etc.
            headers_map (dict): see below key value pairs
                'accept' (list): list of Accept header strings
                'content_type' (list): list of Content-Type header strings
            api_client (ApiClient) api client instance
            callable (function): the function which is invoked when the
                Endpoint is called
        """
        self.settings = settings
        self.params_map = params_map
        self.params_map["all"].extend(
            [
                "async_req",
                "async_threadpool_executor",
                "_host_index",
                "_preload_content",
                "_request_timeout",
                "_return_http_data_only",
                "_check_input_type",
                "_check_return_type",
            ]
        )
        self.params_map["nullable"].extend(["_request_timeout"])
        self.validations = root_map["validations"]
        self.allowed_values = root_map["allowed_values"]
        self.openapi_types = root_map["openapi_types"]
        extra_types = {
            "async_req": (bool,),
            "async_threadpool_executor": (bool,),
            "_host_index": (none_type, int),
            "_preload_content": (bool,),
            "_request_timeout": (none_type, float, (float,), [float], int, (int,), [int]),
            "_return_http_data_only": (bool,),
            "_check_input_type": (bool,),
            "_check_return_type": (bool,),
        }
        self.openapi_types.update(extra_types)
        self.attribute_map = root_map["attribute_map"]
        self.location_map = root_map["location_map"]
        self.collection_format_map = root_map["collection_format_map"]
        self.headers_map = headers_map
        self.api_client = api_client
        self.callable = callable

    def __validate_inputs(self, kwargs):
        for param in self.params_map["enum"]:
            if param in kwargs:
                check_allowed_values(self.allowed_values, (param,), kwargs[param])

        for param in self.params_map["validation"]:
            if param in kwargs:
                check_validations(
                    self.validations,
                    (param,),
                    kwargs[param],
                    configuration=self.api_client.configuration,
                )

        if kwargs["_check_input_type"] is False:
            return

        for key, value in kwargs.items():
            fixed_val = validate_and_convert_types(
                value,
                self.openapi_types[key],
                [key],
                False,
                kwargs["_check_input_type"],
                configuration=self.api_client.configuration,
            )
            kwargs[key] = fixed_val

    def __gather_params(self, kwargs):
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
            param_location = self.location_map.get(param_name)
            if param_location is None:
                continue
            if param_location:
                if param_location == "body":
                    params["body"] = param_value
                    continue
                base_name = self.attribute_map[param_name]
                if param_location == "form" and self.openapi_types[param_name] == (file_type,):
                    params["file"][param_name] = [param_value]
                elif param_location == "form" and self.openapi_types[param_name] == ([file_type],):
                    # param_value is already a list
                    params["file"][param_name] = param_value
                elif param_location in {"form", "query"}:
                    param_value_full = (base_name, param_value)
                    params[param_location].append(param_value_full)
                if param_location not in {"form", "query"}:
                    params[param_location][base_name] = param_value
                collection_format = self.collection_format_map.get(param_name)
                if collection_format:
                    params["collection_format"][base_name] = collection_format

        return params

    def __call__(self, *args, **kwargs):
        """This method is invoked when endpoints are called
        Example:

        api_instance = InferenceApi()
        api_instance.embed  # this is an instance of the class Endpoint
        api_instance.embed()  # this invokes api_instance.embed.__call__()
        which then invokes the callable functions stored in that endpoint at
        api_instance.embed.callable or self.callable in this class

        """
        return self.callable(self, *args, **kwargs)

    def call_with_http_info(self, **kwargs):
        try:
            index = (
                self.api_client.configuration.server_operation_index.get(
                    self.settings["operation_id"], self.api_client.configuration.server_index
                )
                if kwargs["_host_index"] is None
                else kwargs["_host_index"]
            )
            server_variables = self.api_client.configuration.server_operation_variables.get(
                self.settings["operation_id"], self.api_client.configuration.server_variables
            )
            _host = self.api_client.configuration.get_host_from_settings(
                index, variables=server_variables, servers=self.settings["servers"]
            )
        except IndexError:
            if self.settings["servers"]:
                raise PineconeApiValueError(
                    "Invalid host index. Must be 0 <= index < %s" % len(self.settings["servers"])
                )
            _host = None

        for key, value in kwargs.items():
            if key not in self.params_map["all"]:
                raise PineconeApiTypeError(
                    "Got an unexpected parameter '%s'"
                    " to method `%s`" % (key, self.settings["operation_id"])
                )
            # only throw this nullable PineconeApiValueError if _check_input_type
            # is False, if _check_input_type==True we catch this case
            # in self.__validate_inputs
            if (
                key not in self.params_map["nullable"]
                and value is None
                and kwargs["_check_input_type"] is False
            ):
                raise PineconeApiValueError(
                    "Value may not be None for non-nullable parameter `%s`"
                    " when calling `%s`" % (key, self.settings["operation_id"])
                )

        for key in self.params_map["required"]:
            if key not in kwargs.keys():
                raise PineconeApiValueError(
                    "Missing the required parameter `%s` when calling "
                    "`%s`" % (key, self.settings["operation_id"])
                )

        self.__validate_inputs(kwargs)

        params = self.__gather_params(kwargs)

        accept_headers_list = self.headers_map["accept"]
        if accept_headers_list:
            params["header"]["Accept"] = self.api_client.select_header_accept(accept_headers_list)

        content_type_headers_list = self.headers_map["content_type"]
        if content_type_headers_list:
            header_list = self.api_client.select_header_content_type(content_type_headers_list)
            params["header"]["Content-Type"] = header_list

        return self.api_client.call_api(
            self.settings["endpoint_path"],
            self.settings["http_method"],
            params["path"],
            params["query"],
            params["header"],
            body=params["body"],
            post_params=params["form"],
            files=params["file"],
            response_type=self.settings["response_type"],
            auth_settings=self.settings["auth"],
            async_req=kwargs["async_req"],
            async_threadpool_executor=kwargs.get("async_threadpool_executor", None),
            _check_type=kwargs["_check_return_type"],
            _return_http_data_only=kwargs["_return_http_data_only"],
            _preload_content=kwargs["_preload_content"],
            _request_timeout=kwargs["_request_timeout"],
            _host=_host,
            collection_formats=params["collection_format"],
        )
