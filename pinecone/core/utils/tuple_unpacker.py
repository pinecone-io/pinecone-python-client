from typing import Tuple, List, Type, Dict, Any


class TupleUnpacker:

    """this class is used to unpack a tuple into a dictionary of arguments"""

    def __init__(self,
                 ordered_required_items: List[Tuple[str, Type]],
                 ordered_optional_items: List[Tuple[str, Type, Any]]):
        """
        Args:
            ordered_required_items: a list of required tuples of the form (arg_name, arg_type_hint).
            ordered_optional_items: a list of optional tuples of the form (arg_name, arg_type_hint, missing_value).

            Note that optional refers to the tuple and not the output arguments.
        """
        self._ordered_required_items = ordered_required_items
        self._ordered_optional_items = ordered_optional_items

    def unpack(self, tup: Tuple) -> Dict[str, Any]:
        """
        Unpack a tuple into a dictionary of arguments.
        Args:
            tup: tuple object to unpack.

        Returns: a dictionary of arguments. All required and optional arguments are included.
                 While for optional arguments, the missing_value is used if the argument is missing in the input tuple.
        """

        if len(tup) < len(self._ordered_required_items):
            missing_args_and_positions = [(arg_name, position)
                                          for position, arg_name
                                          in enumerate(self._ordered_required_items) if position >= len(tup)]
            message = "Missing required arguments in input tuple.\n"
            for arg_name, position in missing_args_and_positions:
                message += f"Argument '{arg_name}' is missing at position {position}.\n"
            message += f"Input tuple: {tup}"
            raise TypeError(message)

        if len(tup) > len(self._ordered_required_items) + len(self._ordered_optional_items):
            most_expected = len(self._ordered_required_items) + len(self._ordered_optional_items)
            message = f"Too many arguments in input tuple.\n Expected at most: {most_expected}\n"
            message += f"Input tuple: {tup}"
            raise TypeError(message)

        res = {}
        cur_pos = 0
        for arg_name, arg_type in self._ordered_required_items:
            arg = tup[cur_pos]
            if not TupleUnpacker._is_of_type(arg, arg_type):
                raise TypeError(f"Argument '{arg_name}' in position {cur_pos} must be of type: {arg_type}.\n"
                                f"Received value: {arg}")
            res[arg_name] = arg
            cur_pos += 1

        for arg_name, arg_type, missing_value in self._ordered_optional_items:
            if cur_pos < len(tup):
                arg = tup[cur_pos]
                if TupleUnpacker._is_of_type(arg, arg_type):
                    res[arg_name] = arg
                    cur_pos += 1
                else:
                    res[arg_name] = missing_value
            else:
                res[arg_name] = missing_value

        if cur_pos < len(tup):
            raise TypeError(f"Unexpected argument in position {cur_pos}.\n"
                            f"Input tuple: {tup}")

        return res

    @staticmethod
    def _is_of_type(obj, type_hint: Type) -> bool:
        if type_hint in (int, float, str, bool):
            return isinstance(obj, type_hint)
        elif type_hint == List[float]:
            return TupleUnpacker._is_list_of(obj, float)
        elif type_hint == Dict[str, float]:
            return TupleUnpacker._is_dict_of(obj, str, float)
        elif type_hint == Dict[str, Any]:
            return TupleUnpacker._is_dict_of(obj, str, Any)
        elif type_hint == Dict[int, float]:
            return TupleUnpacker._is_dict_of(obj, int, float)
        elif type_hint == Dict[int, int]:
            return TupleUnpacker._is_dict_of(obj, int, int)
        else:
            raise ValueError(f"Unsupported type hint: {type_hint}")

    @staticmethod
    def _is_list_of(obj, cls):
        return isinstance(obj, list) and all(isinstance(x, cls) for x in obj)

    @staticmethod
    def _is_dict_of(obj, key_cls, value_cls):
        return isinstance(obj, dict) and all(
            isinstance(k, key_cls) and (value_cls == Any or isinstance(v, value_cls)) for k, v in obj.items())
