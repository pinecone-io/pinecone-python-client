"""
Pinecone Data Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-01
Contact: support@pinecone.io
"""

from pinecone.openapi_support import (  # noqa: F401
    PineconeApiTypeError,
    ModelComposed,
    ModelNormal,
    ModelSimple,
    cached_property,
    change_keys_js_to_python,
    convert_js_args_to_python_args,
    date,
    datetime,
    file_type,
    none_type,
    validate_get_composed_info,
)
from pinecone.openapi_support.model_utils import OpenApiModel
from pinecone.openapi_support.exceptions import PineconeApiAttributeError


def lazy_import():
    from pinecone.core.openapi.db_data.model.query_vector import QueryVector
    from pinecone.core.openapi.db_data.model.sparse_values import SparseValues

    globals()["QueryVector"] = QueryVector
    globals()["SparseValues"] = SparseValues


from typing import Dict, Literal, Tuple, Set, Any, Type, TypeVar
from pinecone.openapi_support import PropertyValidationTypedDict, cached_class_property

T = TypeVar("T", bound="QueryRequest")


class QueryRequest(ModelNormal):
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.

    Attributes:
      allowed_values (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          with a capitalized key describing the allowed value and an allowed
          value. These dicts store the allowed enum values.
      attribute_map (dict): The key is attribute name
          and the value is json key in definition.
      discriminator_value_class_map (dict): A dict to go from the discriminator
          variable value to the discriminator class name.
      validations (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          that stores validations for max_length, min_length, max_items,
          min_items, exclusive_maximum, inclusive_maximum, exclusive_minimum,
          inclusive_minimum, and regex.
      additional_properties_type (tuple): A tuple of classes accepted
          as additional properties values.
    """

    _data_store: Dict[str, Any]
    _check_type: bool

    allowed_values: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    validations: Dict[Tuple[str, ...], PropertyValidationTypedDict] = {
        ("top_k",): {"inclusive_maximum": 10000, "inclusive_minimum": 1},
        ("queries",): {},
        ("vector",): {},
        ("id",): {"max_length": 512},
    }

    @cached_class_property
    def additional_properties_type(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded
        """
        lazy_import()
        return (bool, dict, float, int, list, str, none_type)  # noqa: E501

    _nullable = False

    @cached_class_property
    def openapi_types(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded

        Returns
            openapi_types (dict): The key is attribute name
                and the value is attribute type.
        """
        lazy_import()
        return {
            "top_k": (int,),  # noqa: E501
            "namespace": (str,),  # noqa: E501
            "filter": ({str: (bool, dict, float, int, list, str, none_type)},),  # noqa: E501
            "include_values": (bool,),  # noqa: E501
            "include_metadata": (bool,),  # noqa: E501
            "queries": ([QueryVector],),  # noqa: E501
            "vector": ([float],),  # noqa: E501
            "sparse_vector": (SparseValues,),  # noqa: E501
            "id": (str,),  # noqa: E501
        }

    @cached_class_property
    def discriminator(cls):
        return None

    attribute_map: Dict[str, str] = {
        "top_k": "topK",  # noqa: E501
        "namespace": "namespace",  # noqa: E501
        "filter": "filter",  # noqa: E501
        "include_values": "includeValues",  # noqa: E501
        "include_metadata": "includeMetadata",  # noqa: E501
        "queries": "queries",  # noqa: E501
        "vector": "vector",  # noqa: E501
        "sparse_vector": "sparseVector",  # noqa: E501
        "id": "id",  # noqa: E501
    }

    read_only_vars: Set[str] = set([])

    _composed_schemas: Dict[Literal["allOf", "oneOf", "anyOf"], Any] = {}

    @classmethod
    @convert_js_args_to_python_args
    def _from_openapi_data(cls: Type[T], top_k, *args, **kwargs) -> T:  # noqa: E501
        """QueryRequest - a model defined in OpenAPI

        Args:
            top_k (int): The number of results to return for each query.

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
            namespace (str): The namespace to query. [optional]  # noqa: E501
            filter ({str: (bool, dict, float, int, list, str, none_type)}): The filter to apply. You can use vector metadata to limit your search. See [Understanding metadata](https://docs.pinecone.io/guides/data/understanding-metadata). [optional]  # noqa: E501
            include_values (bool): Indicates whether vector values are included in the response. [optional] if omitted the server will use the default value of False.  # noqa: E501
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids. [optional] if omitted the server will use the default value of False.  # noqa: E501
            queries ([QueryVector]): DEPRECATED. The query vectors. Each `query()` request can contain only one of the parameters `queries`, `vector`, or  `id`. [optional]  # noqa: E501
            vector ([float]): The query vector. This should be the same length as the dimension of the index being queried. Each `query` request can contain only one of the parameters `id` or `vector`. [optional]  # noqa: E501
            sparse_vector (SparseValues): [optional]  # noqa: E501
            id (str): The unique ID of the vector to be used as a query vector. Each `query` request can contain only one of the parameters `queries`, `vector`, or  `id`. [optional]  # noqa: E501
        """

        _check_type = kwargs.pop("_check_type", True)
        _spec_property_naming = kwargs.pop("_spec_property_naming", False)
        _path_to_item = kwargs.pop("_path_to_item", ())
        _configuration = kwargs.pop("_configuration", None)
        _visited_composed_classes = kwargs.pop("_visited_composed_classes", ())

        self = super(OpenApiModel, cls).__new__(cls)

        if args:
            raise PineconeApiTypeError(
                "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments."
                % (args, self.__class__.__name__),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        self._data_store = {}
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.top_k = top_k
        for var_name, var_value in kwargs.items():
            if (
                var_name not in self.attribute_map
                and self._configuration is not None
                and self._configuration.discard_unknown_keys
                and self.additional_properties_type is None
            ):
                # discard variable.
                continue
            setattr(self, var_name, var_value)
        return self

    required_properties = set(
        [
            "_data_store",
            "_check_type",
            "_spec_property_naming",
            "_path_to_item",
            "_configuration",
            "_visited_composed_classes",
        ]
    )

    @convert_js_args_to_python_args
    def __init__(self, top_k, *args, **kwargs) -> None:  # noqa: E501
        """QueryRequest - a model defined in OpenAPI

        Args:
            top_k (int): The number of results to return for each query.

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
            namespace (str): The namespace to query. [optional]  # noqa: E501
            filter ({str: (bool, dict, float, int, list, str, none_type)}): The filter to apply. You can use vector metadata to limit your search. See [Understanding metadata](https://docs.pinecone.io/guides/data/understanding-metadata). [optional]  # noqa: E501
            include_values (bool): Indicates whether vector values are included in the response. [optional] if omitted the server will use the default value of False.  # noqa: E501
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids. [optional] if omitted the server will use the default value of False.  # noqa: E501
            queries ([QueryVector]): DEPRECATED. The query vectors. Each `query()` request can contain only one of the parameters `queries`, `vector`, or  `id`. [optional]  # noqa: E501
            vector ([float]): The query vector. This should be the same length as the dimension of the index being queried. Each `query` request can contain only one of the parameters `id` or `vector`. [optional]  # noqa: E501
            sparse_vector (SparseValues): [optional]  # noqa: E501
            id (str): The unique ID of the vector to be used as a query vector. Each `query` request can contain only one of the parameters `queries`, `vector`, or  `id`. [optional]  # noqa: E501
        """

        _check_type = kwargs.pop("_check_type", True)
        _spec_property_naming = kwargs.pop("_spec_property_naming", False)
        _path_to_item = kwargs.pop("_path_to_item", ())
        _configuration = kwargs.pop("_configuration", None)
        _visited_composed_classes = kwargs.pop("_visited_composed_classes", ())

        if args:
            raise PineconeApiTypeError(
                "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments."
                % (args, self.__class__.__name__),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        self._data_store = {}
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.top_k = top_k
        for var_name, var_value in kwargs.items():
            if (
                var_name not in self.attribute_map
                and self._configuration is not None
                and self._configuration.discard_unknown_keys
                and self.additional_properties_type is None
            ):
                # discard variable.
                continue
            setattr(self, var_name, var_value)
            if var_name in self.read_only_vars:
                raise PineconeApiAttributeError(
                    f"`{var_name}` is a read-only attribute. Use `from_openapi_data` to instantiate "
                    f"class with read only attributes."
                )
