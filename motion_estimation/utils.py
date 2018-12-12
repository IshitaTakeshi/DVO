def to_tuple_if_scalar(value):
    """
    If a scalar value is given, duplicate it and return as a 2 element tuple.
    """
    if isinstance(value, float) or isinstance(value, int):
        return (value, value)
    return value

