import json
import atexit
import mimetypes
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import io
import os
import re
from urllib.parse import quote
from urllib3.fields import RequestField

from typing import Optional, List, Tuple, Dict, Any, Union


from .rest_urllib3 import Urllib3RestClient
from .configuration import Configuration
from .exceptions import PineconeApiValueError, PineconeApiException
from .model_utils import (
    ModelNormal,
    ModelSimple,
    ModelComposed,
    date,
    datetime,
    deserialize_file,
    file_type,
    model_to_dict,
    validate_and_convert_types,
)


class ApiClient(object):
    """Generic API client for OpenAPI client library builds.

    :param configuration: .Configuration object for this client
    :param header_name: a header to pass when making calls to the API.
    :param header_value: a header value to pass when making calls to
        the API.
    :param cookie: a cookie to include in the header when making calls
        to the API
    :param pool_threads: The number of threads to use for async requests
        to the API. More threads means more concurrent API requests.
    """

    _pool = None
    _threadpool_executor = None

    def __init__(
        self, configuration=None, header_name=None, header_value=None, cookie=None, pool_threads=1
    ):
        if configuration is None:
            configuration = Configuration.get_default_copy()
        self.configuration = configuration
        self.pool_threads = pool_threads

        # self.rest_client = HttpxRestClient(configuration, http2=False)
        self.rest_client = Urllib3RestClient(configuration)

        self.default_headers = {}
        if header_name is not None:
            self.default_headers[header_name] = header_value
        self.cookie = cookie
        # Set default User-Agent.
        self.user_agent = "OpenAPI-Generator/1.0.0/python"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._threadpool_executor:
            self._threadpool_executor.shutdown()
            self._threadpool_executor = None
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None
            if hasattr(atexit, "unregister"):
                atexit.unregister(self.close)

    @property
    def pool(self):
        """Create thread pool on first request
        avoids instantiating unused threadpool for blocking clients.
        """
        if self._pool is None:
            atexit.register(self.close)
            self._pool = ThreadPool(self.pool_threads)
        return self._pool

    @property
    def threadpool_executor(self):
        if self._threadpool_executor is None:
            self._threadpool_executor = ThreadPoolExecutor(max_workers=self.pool_threads)
        return self._threadpool_executor

    @property
    def user_agent(self):
        """User agent for this API client"""
        return self.default_headers["User-Agent"]

    @user_agent.setter
    def user_agent(self, value):
        self.default_headers["User-Agent"] = value

    def set_default_header(self, header_name, header_value):
        self.default_headers[header_name] = header_value

    def __call_api(
        self,
        resource_path: str,
        method: str,
        path_params: Optional[Dict[str, Any]] = None,
        query_params: Optional[List[Tuple[str, Any]]] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        post_params: Optional[List[Tuple[str, Any]]] = None,
        files: Optional[Dict[str, List[io.IOBase]]] = None,
        response_type: Optional[Tuple[Any]] = None,
        auth_settings: Optional[List[str]] = None,
        _return_http_data_only: Optional[bool] = None,
        collection_formats: Optional[Dict[str, str]] = None,
        _preload_content: bool = True,
        _request_timeout: Optional[Union[int, float, Tuple]] = None,
        _host: Optional[str] = None,
        _check_type: Optional[bool] = None,
    ):
        config = self.configuration

        # header parameters
        header_params = header_params or {}
        header_params.update(self.default_headers)
        if self.cookie:
            header_params["Cookie"] = self.cookie
        if header_params:
            sanitized_header_params: Dict[str, Any] = self.sanitize_for_serialization(header_params)
            processed_header_params: Dict[str, Any] = dict(
                self.parameters_to_tuples(sanitized_header_params, collection_formats)
            )

        # path parameters
        sanitized_path_params: Dict[str, Any] = self.sanitize_for_serialization(path_params or {})
        processed_path_params: List[Tuple[str, Any]] = self.parameters_to_tuples(
            sanitized_path_params, collection_formats
        )

        for k, v in processed_path_params:
            # specified safe chars, encode everything
            resource_path = resource_path.replace(
                "{%s}" % k, quote(str(v), safe=config.safe_chars_for_path_param)
            )

        # query parameters
        if query_params:
            sanitized_query_params = self.sanitize_for_serialization(query_params)
            processed_query_params = self.parameters_to_tuples(
                sanitized_query_params, collection_formats
            )
        else:
            processed_query_params = []

        # post parameters
        if post_params or files:
            post_params = post_params if post_params else []
            sanitized_post_params = self.sanitize_for_serialization(post_params)
            if sanitized_path_params:
                processed_post_params = self.parameters_to_tuples(
                    sanitized_post_params, collection_formats
                )
                processed_post_params.extend(self.files_parameters(files))
            if processed_header_params["Content-Type"].startswith("multipart"):
                processed_post_params = self.parameters_to_multipart(sanitized_post_params, (dict))
        else:
            processed_post_params = None

        # body
        if body:
            body = self.sanitize_for_serialization(body)

        # auth setting
        self.update_params_for_auth(
            processed_header_params,
            processed_query_params,
            auth_settings,
            resource_path,
            method,
            body,
        )

        # request url
        if _host is None:
            url = self.configuration.host + resource_path
        else:
            # use server/host defined in path or operation instead
            url = _host + resource_path

        try:
            # perform request and return response
            response_data = self.request(
                method,
                url,
                query_params=processed_query_params,
                headers=processed_header_params,
                post_params=processed_post_params,
                body=body,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
            )
        except PineconeApiException as e:
            e.body = e.body.decode("utf-8")
            raise e

        self.last_response = response_data

        return_data = response_data

        if not _preload_content:
            return return_data

        # deserialize response data
        if response_type:
            if response_type != (file_type,):
                encoding = "utf-8"
                content_type = response_data.getheader("content-type")
                if content_type is not None:
                    match = re.search(r"charset=([a-zA-Z\-\d]+)[\s\;]?", content_type)
                    if match:
                        encoding = match.group(1)
                response_data.data = response_data.data.decode(encoding)

            return_data = self.deserialize(response_data, response_type, _check_type)
        else:
            return_data = None

        if _return_http_data_only:
            return return_data
        else:
            return (return_data, response_data.status, response_data.getheaders())

    def parameters_to_multipart(self, params, collection_types):
        """Get parameters as list of tuples, formatting as json if value is collection_types

        :param params: Parameters as list of two-tuples
        :param dict collection_types: Parameter collection types
        :return: Parameters as list of tuple or urllib3.fields.RequestField
        """
        new_params = []
        if collection_types is None:
            collection_types = dict
        for k, v in params.items() if isinstance(params, dict) else params:  # noqa: E501
            if isinstance(
                v, collection_types
            ):  # v is instance of collection_type, formatting as application/json
                v = json.dumps(v, ensure_ascii=False).encode("utf-8")
                field = RequestField(k, v)
                field.make_multipart(content_type="application/json; charset=utf-8")
                new_params.append(field)
            else:
                new_params.append((k, v))
        return new_params

    @classmethod
    def sanitize_for_serialization(cls, obj) -> Any:
        """Prepares data for transmission before it is sent with the rest client
        If obj is None, return None.
        If obj is str, int, long, float, bool, return directly.
        If obj is datetime.datetime, datetime.date
            convert to string in iso8601 format.
        If obj is list, sanitize each element in the list.
        If obj is dict, return the dict.
        If obj is OpenAPI model, return the properties dict.
        If obj is io.IOBase, return the bytes
        :param obj: The data to serialize.
        :return: The serialized form of data.
        """
        if isinstance(obj, (ModelNormal, ModelComposed)):
            return {
                key: cls.sanitize_for_serialization(val)
                for key, val in model_to_dict(obj, serialize=True).items()
            }
        elif isinstance(obj, io.IOBase):
            return cls.get_file_data_and_close_file(obj)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, ModelSimple):
            return cls.sanitize_for_serialization(obj.value)
        elif isinstance(obj, (list, tuple)):
            return [cls.sanitize_for_serialization(item) for item in obj]
        if isinstance(obj, dict):
            return {key: cls.sanitize_for_serialization(val) for key, val in obj.items()}
        raise PineconeApiValueError(
            "Unable to prepare type {} for serialization".format(obj.__class__.__name__)
        )

    def deserialize(self, response, response_type, _check_type):
        """Deserializes response into an object.

        :param response: RESTResponse object to be deserialized.
        :param response_type: For the response, a tuple containing:
            valid classes
            a list containing valid classes (for list schemas)
            a dict containing a tuple of valid classes as the value
            Example values:
            (str,)
            (Pet,)
            (float, none_type)
            ([int, none_type],)
            ({str: (bool, str, int, float, date, datetime, str, none_type)},)
        :param _check_type: boolean, whether to check the types of the data
            received from the server
        :type _check_type: bool

        :return: deserialized object.
        """
        # handle file downloading
        # save response body into a tmp file and return the instance
        if response_type == (file_type,):
            content_disposition = response.getheader("Content-Disposition")
            return deserialize_file(
                response.data, self.configuration, content_disposition=content_disposition
            )

        # fetch data from response object
        try:
            received_data = json.loads(response.data)
        except ValueError:
            received_data = response.data

        # store our data under the key of 'received_data' so users have some
        # context if they are deserializing a string and the data type is wrong
        deserialized_data = validate_and_convert_types(
            received_data,
            response_type,
            ["received_data"],
            True,
            _check_type,
            configuration=self.configuration,
        )
        return deserialized_data

    def call_api(
        self,
        resource_path: str,
        method: str,
        path_params: Optional[Dict[str, Any]] = None,
        query_params: Optional[List[Tuple[str, Any]]] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        post_params: Optional[List[Tuple[str, Any]]] = None,
        files: Optional[Dict[str, List[io.IOBase]]] = None,
        response_type: Optional[Tuple[Any]] = None,
        auth_settings: Optional[List[str]] = None,
        async_req: Optional[bool] = None,
        async_threadpool_executor: Optional[bool] = None,
        _return_http_data_only: Optional[bool] = None,
        collection_formats: Optional[Dict[str, str]] = None,
        _preload_content: bool = True,
        _request_timeout: Optional[Union[int, float, Tuple]] = None,
        _host: Optional[str] = None,
        _check_type: Optional[bool] = None,
    ):
        """Makes the HTTP request (synchronous) and returns deserialized data.

        To make an async_req request, set the async_req parameter.

        :param resource_path: Path to method endpoint.
        :param method: Method to call.
        :param path_params: Path parameters in the url.
        :param query_params: Query parameters in the url.
        :param header_params: Header parameters to be
            placed in the request header.
        :param body: Request body.
        :param post_params dict: Request post form parameters,
            for `application/x-www-form-urlencoded`, `multipart/form-data`.
        :param auth_settings list: Auth Settings names for the request.
        :param response_type: For the response, a tuple containing:
            valid classes
            a list containing valid classes (for list schemas)
            a dict containing a tuple of valid classes as the value
            Example values:
            (str,)
            (Pet,)
            (float, none_type)
            ([int, none_type],)
            ({str: (bool, str, int, float, date, datetime, str, none_type)},)
        :param files: key -> field name, value -> a list of open file
            objects for `multipart/form-data`.
        :type files: dict
        :param async_req bool: execute request asynchronously
        :type async_req: bool, optional
        :param _return_http_data_only: response data without head status code
                                       and headers
        :type _return_http_data_only: bool, optional
        :param collection_formats: dict of collection formats for path, query,
            header, and post parameters.
        :type collection_formats: dict, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _check_type: boolean describing if the data back from the server
            should have its type checked.
        :type _check_type: bool, optional
        :return:
            If async_req parameter is True,
            the request will be called asynchronously.
            The method will return the request thread.
            If parameter async_req is False or missing,
            then the method will return the response directly.
        """
        if async_threadpool_executor:
            return self.threadpool_executor.submit(
                self.__call_api,
                resource_path,
                method,
                path_params,
                query_params,
                header_params,
                body,
                post_params,
                files,
                response_type,
                auth_settings,
                _return_http_data_only,
                collection_formats,
                _preload_content,
                _request_timeout,
                _host,
                _check_type,
            )

        if not async_req:
            return self.__call_api(
                resource_path,
                method,
                path_params,
                query_params,
                header_params,
                body,
                post_params,
                files,
                response_type,
                auth_settings,
                _return_http_data_only,
                collection_formats,
                _preload_content,
                _request_timeout,
                _host,
                _check_type,
            )

        return self.pool.apply_async(
            self.__call_api,
            (
                resource_path,
                method,
                path_params,
                query_params,
                header_params,
                body,
                post_params,
                files,
                response_type,
                auth_settings,
                _return_http_data_only,
                collection_formats,
                _preload_content,
                _request_timeout,
                _host,
                _check_type,
            ),
        )

    def request(
        self,
        method,
        url,
        query_params=None,
        headers=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        """Makes the HTTP request using RESTClient."""
        if method == "GET":
            return self.rest_client.GET(
                url,
                query_params=query_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                headers=headers,
            )
        elif method == "HEAD":
            return self.rest_client.HEAD(
                url,
                query_params=query_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                headers=headers,
            )
        elif method == "OPTIONS":
            return self.rest_client.OPTIONS(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "POST":
            return self.rest_client.POST(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "PUT":
            return self.rest_client.PUT(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "PATCH":
            return self.rest_client.PATCH(
                url,
                query_params=query_params,
                headers=headers,
                post_params=post_params,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        elif method == "DELETE":
            return self.rest_client.DELETE(
                url,
                query_params=query_params,
                headers=headers,
                _preload_content=_preload_content,
                _request_timeout=_request_timeout,
                body=body,
            )
        else:
            raise PineconeApiValueError(
                "http method must be `GET`, `HEAD`, `OPTIONS`, `POST`, `PATCH`, `PUT` or `DELETE`."
            )

    @classmethod
    def parameters_to_tuples(
        cls,
        params: Union[Dict[str, Any], List[Tuple[str, Any]]],
        collection_formats: Optional[Dict[str, str]],
    ) -> List[Tuple[str, str]]:
        """Get parameters as list of tuples, formatting collections.

        :param params: Parameters as dict or list of two-tuples
        :param dict collection_formats: Parameter collection formats
        :return: Parameters as list of tuples, collections formatted
        """
        new_params: List[Tuple[str, Any]] = []
        if collection_formats is None:
            collection_formats = {}
        for k, v in params.items() if isinstance(params, dict) else params:  # noqa: E501
            if k in collection_formats:
                collection_format = collection_formats[k]
                if collection_format == "multi":
                    new_params.extend((k, value) for value in v)
                else:
                    if collection_format == "ssv":
                        delimiter = " "
                    elif collection_format == "tsv":
                        delimiter = "\t"
                    elif collection_format == "pipes":
                        delimiter = "|"
                    else:  # csv is the default
                        delimiter = ","
                    new_params.append((k, delimiter.join(str(value) for value in v)))
            else:
                new_params.append((k, v))
        return new_params

    @staticmethod
    def get_file_data_and_close_file(file_instance: io.IOBase) -> bytes:
        file_data = file_instance.read()
        file_instance.close()
        return file_data

    def files_parameters(self, files: Optional[Dict[str, List[io.IOBase]]] = None):
        """Builds form parameters.

        :param files: None or a dict with key=param_name and
            value is a list of open file objects
        :return: List of tuples of form parameters with file data
        """
        if files is None:
            return []

        params = []
        for param_name, file_instances in files.items():
            if file_instances is None:
                # if the file field is nullable, skip None values
                continue
            for file_instance in file_instances:
                if file_instance is None:
                    # if the file field is nullable, skip None values
                    continue
                if file_instance.closed is True:
                    raise PineconeApiValueError(
                        "Cannot read a closed file. The passed in file_type "
                        "for %s must be open." % param_name
                    )
                filename = os.path.basename(file_instance.name)  # type: ignore
                filedata = self.get_file_data_and_close_file(file_instance)
                mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                params.append(tuple([param_name, tuple([filename, filedata, mimetype])]))

        return params

    def select_header_accept(self, accepts):
        """Returns `Accept` based on an array of accepts provided.

        :param accepts: List of headers.
        :return: Accept (e.g. application/json).
        """
        if not accepts:
            return

        accepts = [x.lower() for x in accepts]

        if "application/json" in accepts:
            return "application/json"
        else:
            return ", ".join(accepts)

    def select_header_content_type(self, content_types: List[str]) -> str:
        """Returns `Content-Type` based on an array of content_types provided.

        :param content_types: List of content-types.
        :return: Content-Type (e.g. application/json).
        """
        if not content_types:
            return "application/json"

        content_types = [x.lower() for x in content_types]

        if "application/json" in content_types or "*/*" in content_types:
            return "application/json"
        else:
            return content_types[0]

    def update_params_for_auth(self, headers, querys, auth_settings, resource_path, method, body):
        """Updates header and query params based on authentication setting.

        :param headers: Header parameters dict to be updated.
        :param querys: Query parameters tuple list to be updated.
        :param auth_settings: Authentication setting identifiers list.
        :param resource_path: A string representation of the HTTP request resource path.
        :param method: A string representation of the HTTP request method.
        :param body: A object representing the body of the HTTP request.
            The object type is the return value of _encoder.default().
        """
        if not auth_settings:
            return

        for auth in auth_settings:
            auth_setting = self.configuration.auth_settings().get(auth)
            if auth_setting:
                if auth_setting["in"] == "cookie":
                    headers["Cookie"] = auth_setting["value"]
                elif auth_setting["in"] == "header":
                    if auth_setting["type"] != "http-signature":
                        headers[auth_setting["key"]] = auth_setting["value"]
                elif auth_setting["in"] == "query":
                    querys.append((auth_setting["key"], auth_setting["value"]))
                else:
                    raise PineconeApiValueError(
                        "Authentication token must be in `query` or `header`"
                    )
