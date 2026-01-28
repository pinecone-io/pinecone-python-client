"""
Pinecone Control Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors and documents.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2026-01.alpha
Contact: support@pinecone.io
"""

from pinecone.openapi_support.model_utils import (  # noqa: F401
    PineconeApiTypeError,
    ModelComposed,
    ModelNormal,
    ModelSimple,
    OpenApiModel,
    cached_property,
    change_keys_js_to_python,
    convert_js_args_to_python_args,
    date,
    datetime,
    file_type,
    none_type,
    validate_get_composed_info,
)
from pinecone.openapi_support.exceptions import PineconeApiAttributeError


from typing import Dict, Literal, Tuple, Set, Any, Type, TypeVar
from pinecone.openapi_support import PropertyValidationTypedDict, cached_class_property

T = TypeVar("T", bound="SchemaFields")


class SchemaFields(ModelNormal):
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
        ("dimension",): {"inclusive_maximum": 20000, "inclusive_minimum": 1}
    }

    @cached_class_property
    def additional_properties_type(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded
        """
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
        return {
            "type": (str,),  # noqa: E501
            "description": (str,),  # noqa: E501
            "filterable": (bool,),  # noqa: E501
            "full_text_searchable": (bool,),  # noqa: E501
            "dimension": (int,),  # noqa: E501
            "metric": (str,),  # noqa: E501
            "model": (str,),  # noqa: E501
            "field_map": ({str: (str,)},),  # noqa: E501
            "read_parameters": (Dict[str, Any],),  # noqa: E501
            "write_parameters": (Dict[str, Any],),  # noqa: E501
        }

    @cached_class_property
    def discriminator(cls):
        return None

    attribute_map: Dict[str, str] = {
        "type": "type",  # noqa: E501
        "description": "description",  # noqa: E501
        "filterable": "filterable",  # noqa: E501
        "full_text_searchable": "full_text_searchable",  # noqa: E501
        "dimension": "dimension",  # noqa: E501
        "metric": "metric",  # noqa: E501
        "model": "model",  # noqa: E501
        "field_map": "field_map",  # noqa: E501
        "read_parameters": "read_parameters",  # noqa: E501
        "write_parameters": "write_parameters",  # noqa: E501
    }

    read_only_vars: Set[str] = set([])

    _composed_schemas: Dict[Literal["allOf", "oneOf", "anyOf"], Any] = {}

    def __new__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Create a new instance of SchemaFields.

        This method is overridden to provide proper type inference for mypy.
        The actual instance creation logic (including discriminator handling)
        is handled by the parent class's __new__ method.
        """
        # Call parent's __new__ with all arguments to preserve discriminator logic
        instance: T = super().__new__(cls, *args, **kwargs)
        return instance

    @classmethod
    @convert_js_args_to_python_args
    def _from_openapi_data(cls: Type[T], type, *args, **kwargs) -> T:  # noqa: E501
        """SchemaFields - a model defined in OpenAPI

        Args:
            type (str): The data type of the field. Can be a base type (string, integer) or a vector type (dense_vector, sparse_vector, semantic_text).

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
            description (str): A description of the field. [optional]  # noqa: E501
            filterable (bool): Whether the field is filterable. If true, the field is indexed and can be used in query filters. Only applicable for base types (string, integer). [optional]  # noqa: E501
            full_text_searchable (bool): Whether the field is full-text searchable. If true, the field is indexed for lexical search. Only applicable for string type fields. [optional]  # noqa: E501
            dimension (int): The dimension of the dense vectors. Required when type is dense_vector. [optional]  # noqa: E501
            metric (str): The distance metric to be used for similarity search. Required when type is dense_vector or sparse_vector. Optional when type is semantic_text (may be included in responses). For dense_vector: cosine, euclidean, or dotproduct. For sparse_vector: must be dotproduct. For semantic_text: typically cosine. [optional]  # noqa: E501
            model (str): The name of the embedding model to use. Required when type is semantic_text. [optional]  # noqa: E501
            field_map ({str: (str,)}): Identifies the name of the text field from your document model that will be embedded. Maps the field name in your documents to the field name used for embedding. Only applicable when type is semantic_text. [optional]  # noqa: E501
            read_parameters (Dict[str, Any]): The read parameters for the embedding model used during queries. Only applicable when type is semantic_text. [optional]  # noqa: E501
            write_parameters (Dict[str, Any]): The write parameters for the embedding model used during indexing. Only applicable when type is semantic_text. [optional]  # noqa: E501
        """

        _enforce_allowed_values = kwargs.pop("_enforce_allowed_values", False)
        _enforce_validations = kwargs.pop("_enforce_validations", False)
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
        self._enforce_allowed_values = _enforce_allowed_values
        self._enforce_validations = _enforce_validations
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.type = type
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
            "_enforce_allowed_values",
            "_enforce_validations",
            "_data_store",
            "_check_type",
            "_spec_property_naming",
            "_path_to_item",
            "_configuration",
            "_visited_composed_classes",
        ]
    )

    @convert_js_args_to_python_args
    def __init__(self, type, *args, **kwargs) -> None:  # noqa: E501
        """SchemaFields - a model defined in OpenAPI

        Args:
            type (str): The data type of the field. Can be a base type (string, integer) or a vector type (dense_vector, sparse_vector, semantic_text).

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
            description (str): A description of the field. [optional]  # noqa: E501
            filterable (bool): Whether the field is filterable. If true, the field is indexed and can be used in query filters. Only applicable for base types (string, integer). [optional]  # noqa: E501
            full_text_searchable (bool): Whether the field is full-text searchable. If true, the field is indexed for lexical search. Only applicable for string type fields. [optional]  # noqa: E501
            dimension (int): The dimension of the dense vectors. Required when type is dense_vector. [optional]  # noqa: E501
            metric (str): The distance metric to be used for similarity search. Required when type is dense_vector or sparse_vector. Optional when type is semantic_text (may be included in responses). For dense_vector: cosine, euclidean, or dotproduct. For sparse_vector: must be dotproduct. For semantic_text: typically cosine. [optional]  # noqa: E501
            model (str): The name of the embedding model to use. Required when type is semantic_text. [optional]  # noqa: E501
            field_map ({str: (str,)}): Identifies the name of the text field from your document model that will be embedded. Maps the field name in your documents to the field name used for embedding. Only applicable when type is semantic_text. [optional]  # noqa: E501
            read_parameters (Dict[str, Any]): The read parameters for the embedding model used during queries. Only applicable when type is semantic_text. [optional]  # noqa: E501
            write_parameters (Dict[str, Any]): The write parameters for the embedding model used during indexing. Only applicable when type is semantic_text. [optional]  # noqa: E501
        """

        _enforce_allowed_values = kwargs.pop("_enforce_allowed_values", True)
        _enforce_validations = kwargs.pop("_enforce_validations", True)
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
        self._enforce_allowed_values = _enforce_allowed_values
        self._enforce_validations = _enforce_validations
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.type = type
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
