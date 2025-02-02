from typing import Union, Dict, Any, List, Tuple, Optional


def parameters_to_tuples(
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
