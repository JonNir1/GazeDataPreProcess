import inspect


def get_property(obj, attr: str):
    """
    Returns the value of `attr` if it is a public property of `obj`, otherwise raises an AttributeError.
    see https://stackoverflow.com/a/9058322/8543025
    """
    attributes = inspect.getmembers(obj, lambda a: not (inspect.isroutine(a)))
    public_attributes = [a for a in attributes if not a[0].startswith('_')]
    if not (attr in [a[0] for a in public_attributes]):
        raise AttributeError(f"{type(obj).capitalize()} objects have no attribute `{attr}`")
    return getattr(obj, attr)

