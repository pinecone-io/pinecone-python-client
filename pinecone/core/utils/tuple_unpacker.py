from typing import Tuple, List, Type, Dict, Any


class TupleUnpacker:

    @staticmethod
    def unpack(tup: Tuple,
               ordered_required_items: List[Tuple[str, Type]],
               ordered_optional_items: List[Tuple[str, Type]]) -> Tuple:
        if len(tup) < len(ordered_required_items):
            missing_args_and_positions = [(arg_name, position)
                                          for position, arg_name
                                          in enumerate(ordered_required_items) if position >= len(tup)]
            message = "Missing required arguments in input tuple.\n"
            for arg_name, position in missing_args_and_positions:
                message += f"Argument '{arg_name}' is missing at position {position}.\n"
            message += f"Input tuple: {tup}"
            raise TypeError(message)

        if len(tup) > len(ordered_required_items) + len(ordered_optional_items):
            most_expected = len(ordered_required_items) + len(ordered_optional_items)
            message = f"Too many arguments in input tuple.\n Expected at most: {most_expected}\n"
            message += f"Input tuple: {tup}"
            raise TypeError(message)

        res = []
        cur_pos = 0
        for arg_name, arg_type in ordered_required_items:
            arg = tup[cur_pos]
            if not TupleUnpacker._is_of_type(arg, arg_type):
                raise TypeError(f"Argument '{arg_name}' in position {cur_pos} must be of type: {arg_type}.\n"
                                f"Received value: {arg}")
            res.append(arg)
            cur_pos += 1

        for arg_name, arg_type in ordered_optional_items:
            if cur_pos < len(tup):
                arg = tup[cur_pos]
                if TupleUnpacker._is_of_type(arg, arg_type):
                    res.append(arg)
                    cur_pos += 1
                else:
                    res.append(None)
            else:
                res.append(None)

        if cur_pos < len(tup):
            raise TypeError(f"Unexpected argument in position {cur_pos}.\n"
                            f"Input tuple: {tup}")

        return tuple(res)

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
