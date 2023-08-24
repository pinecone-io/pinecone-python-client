
def check_kwargs(caller, given):
    import inspect
    argspec = inspect.getfullargspec(caller)
    diff = set(given).difference(argspec.args)
    if diff:
        raise TypeError(caller.__name__ + ' had unexpected keyword argument(s): ' + ', '.join(diff))